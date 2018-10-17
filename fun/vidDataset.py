import os
import sys
sys.path.append('..')
sys.path.append('../annotations')
sys.path.append('../../annotations')
import torch.utils.data as data
import cv2
import numpy as np
from util.mytoolbox import *
import random
import scipy.io as sio
import copy
import torch
from util.get_image_size import get_image_size
from evalDet import *
from datasetParser import extAllFrmFn
import pdb
from netUtil import *
from wsParamParser import parse_args
from ptd_api import *
from vidDatasetParser import *
from multiprocessing import Process, Pipe, cpu_count, Queue
from vidDatasetParser import vidInfoParser
from multiGraphAttention import extract_position_embedding 
import h5py


class vidDataloader(data.Dataset):
    def __init__(self, ann_folder, prp_type, set_name, dictFile, tubePath, ftrPath, out_cached_folder):
        self.set_name = set_name
        self.dict = pickleload(dictFile)
        self.rpNum = 30
        self.maxWordNum =20
        self.maxTubelegth = 20
        self.tube_ftr_dim = 2048
        self.tubePath = tubePath
        self.ftrPath = ftrPath
        self.out_cache_folder = out_cached_folder
        self.prp_type = prp_type
        self.vid_parser = vidInfoParser(set_name, ann_folder) 
        self.use_key_index = self.vid_parser.tube_cap_dict.keys()
        self.use_key_index.sort()
        self.online_cache ={}
        self.i3d_cache_flag = False
        self.cache_flag = False
        self.cache_ftr_dict = {}
        self.use_mean_cache_flag = False
        self.mean_cache_ftr_path = ''


    def image_samper_set_up(self, rpNum=20, capNum=1, maxWordNum=20, usedBadWord=False, pos_emb_dim=64, pos_type='aiayn', half_size=False, vis_ftr_type='rgb', i3d_ftr_path='', use_mean_cache_flag=False, mean_cache_ftr_path=''):
        self.rpNum = rpNum
        self.maxWordNum = maxWordNum
        self.usedBadWord = usedBadWord
        self.capNum = capNum
        self.pos_emb_dim = pos_emb_dim
        self.pos_type = pos_type
        self.half_size = half_size
        self.vis_ftr_type = vis_ftr_type
        self.i3d_ftr_path = i3d_ftr_path
        self.use_mean_cache_flag = use_mean_cache_flag
        self.mean_cache_ftr_path = mean_cache_ftr_path
        if self.vis_ftr_type=='i3d':
            self.tube_ftr_dim =1024 # 1024 for rgb, 1024 for flow
        self.tube_ftr_dim_i3d =1024 # 1024 for rgb, 1024 for flow

    def __len__(self):
        #return 2000
        return len(self.vid_parser.tube_cap_dict)

    def get_word_emb_from_str(self, capString, maxWordNum):
        capList = caption_to_word_list(capString) 
        wordEmbMatrix= np.zeros((self.maxWordNum, 300), dtype=np.float32)         
        valCount =0
        wordLbl = list()
        for i, word in enumerate(capList):
            if (not self.usedBadWord) and word in self.dict['out_voca']:
                #print('Invalid word: ',  idx, word, len(self.dict['word2vec']))
                continue
            if(valCount>=self.maxWordNum):
                break
            idx = self.dict['word2idx'][word]
            wordEmbMatrix[valCount, :]= self.dict['word2vec'][idx]
            valCount +=1
            wordLbl.append(idx)
        return wordEmbMatrix, valCount

    def get_cap_emb(self, index, capNum):
        cap_list_index = self.vid_parser.tube_cap_dict[index]
        assert len(cap_list_index)>=capNum
        cap_sample_index = random.sample(range(len(cap_list_index)), capNum)

        # get word embedding
        wordEmbMatrix= np.zeros((capNum, self.maxWordNum, 300), dtype=np.float32)
        cap_length_list = list()
        for i, capIdx in enumerate(cap_sample_index):
            capString = cap_list_index[capIdx]
            wordEmbMatrix[i, ...], valid_length = self.get_word_emb_from_str(capString, self.maxWordNum)
            cap_length_list.append(valid_length)
        return wordEmbMatrix, cap_length_list

    def get_tube_embedding(self, index, maxTubelegth, out_cached_folder = ''):
        tube_embedding = np.zeros((self.rpNum, maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
        set_name = self.set_name
        ins_ann, vd_name = self.vid_parser.get_shot_anno_from_index(index)
        tube_info_path = os.path.join(self.tubePath, set_name, self.prp_type, str(index)+'.pd') 
        tubeInfo = pickleload(tube_info_path)
        tube_list, frame_list = tubeInfo
        frmNum = len(frame_list)
        seg_length = max(int(frmNum/maxTubelegth), 1)
        
        tube_to_prp_idx = list()
        ftr_tube_list = list()
        prp_range_num = len(tube_list[0])
        #pdb.set_trace()
        tmp_cache_tube_feature_path = os.path.join(out_cached_folder, \
                    set_name, self.prp_type, str(index) + '.pk')
        if os.path.isfile(tmp_cache_tube_feature_path):
            tmp_tube_ftr_info = None
            try:
                tmp_tube_ftr_info = pickleload(tmp_cache_tube_feature_path)
            except:
                print('--------------------------------------------------')
                print(tmp_cache_tube_feature_path)
                print('--------------------------------------------------')
            if tmp_tube_ftr_info is not None: 
                tube_embedding, tubeInfo, tube_to_prp_idx = tmp_tube_ftr_info

                #if((tube_to_prp_idx[0])>maxTubelegth):
                if tube_embedding.shape[0]>=self.rpNum:
                    return tube_embedding[:self.rpNum], tubeInfo, tube_to_prp_idx
                else:
                    tube_embedding = np.zeros((self.rpNum, maxTubelegth, self.tube_ftr_dim), dtype=np.float32)

        # cache data for saving IO time
        cache_data_dict ={}

        for tubeId, tube in enumerate(tube_list[0]):
            if tubeId>= self.rpNum:
                continue
            tube_prp_map = list()
            # find proposals
            for frmId, bbox in enumerate(tube):
                frmName = frame_list[frmId] 
                if frmName in cache_data_dict.keys():
                    img_prp_ftr_info = cache_data_dict[frmName]
                else:
                    img_prp_ftr_info_path = os.path.join(self.ftrPath, self.set_name, vd_name, frmName+ '.pd')
                    img_prp_ftr_info = pickleload(img_prp_ftr_info_path) 
                    cache_data_dict[frmName] = img_prp_ftr_info
                
                tmp_bbx = copy.deepcopy(img_prp_ftr_info['rois'][:prp_range_num]) # to be modified
                tmp_info = img_prp_ftr_info['imFo'].squeeze()
                tmp_bbx[:, 0] = tmp_bbx[:, 0]/tmp_info[1]
                tmp_bbx[:, 2] = tmp_bbx[:, 2]/tmp_info[1]
                tmp_bbx[:, 1] = tmp_bbx[:, 1]/tmp_info[0]
                tmp_bbx[:, 3] = tmp_bbx[:, 3]/tmp_info[0]
                img_prp_res = tmp_bbx - bbox
                img_prp_res_sum = np.sum(img_prp_res, axis=1)
                for prpId in range(prp_range_num):
                    if(abs(img_prp_res_sum[prpId])<0.00001):
                        tube_prp_map.append(prpId)
                        break
                #assert("fail to find proposals")
            if (len(tube_prp_map)!=len(tube)):
                pdb.set_trace()
            assert(len(tube_prp_map)==len(tube))
           
            tube_to_prp_idx.append(tube_prp_map)
            
            # extract features
            tmp_tube_embedding = np.zeros((maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
            for segId in range(maxTubelegth):
                start_id = segId*seg_length
                end_id = (segId+1)*seg_length
                if end_id>frmNum and frmNum<maxTubelegth:
                    break
                end_id = min((segId+1)*(seg_length), frmNum)
                tmp_ftr = np.zeros((1, self.tube_ftr_dim), dtype=np.float32)
                for frmId in range(start_id, end_id):
                    frm_name = frame_list[frmId]
                    if frm_name in cache_data_dict.keys():
                        img_prp_ftr_info = cache_data_dict[frm_name]
                    else:
                        img_prp_ftr_info_path = os.path.join(self.ftrPath, self.set_name, vd_name, frm_name+ '.pd')
                        img_prp_ftr_info = pickleload(img_prp_ftr_info_path) 
                        cache_data_dict[frm_name] = img_prp_ftr_info
                    tmp_ftr +=img_prp_ftr_info['roiFtr'][tube_prp_map[frmId]]
                tmp_tube_embedding[segId, :] = tmp_ftr/(end_id-start_id)
            
            tube_embedding[tubeId, ...] = tmp_tube_embedding
        
        if out_cached_folder !='':
            dir_name = os.path.dirname(tmp_cache_tube_feature_path)
            makedirs_if_missing(dir_name)
            pickledump(tmp_cache_tube_feature_path, [tube_embedding, tubeInfo, tube_to_prp_idx])
    
        if self.half_size:
            tube_embedding = tube_embedding.view(self.rpNum, self.maxTubelegth/2, 2, self.tube_ftr_dim)
            tube_embedding = np.mean(tube_embedding, axis=2)

        return tube_embedding, tubeInfo, tube_to_prp_idx


    def get_tube_embedding_i3d(self, index, maxTubelegth, out_cached_folder = ''):
        #pdb.set_trace()
        rgb_tube_embedding = np.zeros((self.rpNum, maxTubelegth, self.tube_ftr_dim_i3d), dtype=np.float32)
        flow_tube_embedding = np.zeros((self.rpNum, maxTubelegth, self.tube_ftr_dim_i3d), dtype=np.float32)
        set_name = self.set_name
        i3d_ftr_path =  os.path.join(self.i3d_ftr_path, set_name, str(index) +'.h5')
        if i3d_ftr_path in self.online_cache.keys() and self.i3d_cache_flag:
            tube_embedding = self.online_cache[i3d_ftr_path]
            return tube_embedding
        h5_handle = h5py.File(i3d_ftr_path, 'r')
        for tube_id in range(self.rpNum):
            rgb_tube_ftr = h5_handle[str(tube_id)]['rgb_feature'][()].squeeze()
            flow_tube_ftr = h5_handle[str(tube_id)]['flow_feature'][()].squeeze()
            num_tube_ftr = h5_handle[str(tube_id)]['num_feature'][()].squeeze()
            seg_length = max(int(round(num_tube_ftr/maxTubelegth)), 1)
            tmp_rgb_tube_embedding = np.zeros((maxTubelegth, self.tube_ftr_dim_i3d), dtype=np.float32)
            tmp_flow_tube_embedding = np.zeros((maxTubelegth, self.tube_ftr_dim_i3d), dtype=np.float32)
            #pdb.set_trace()
            for segId in range(maxTubelegth):
                #print('%d %d\n' %(tube_id, segId))
                start_id = segId*seg_length
                end_id = (segId+1)*seg_length
                if end_id > num_tube_ftr and num_tube_ftr < maxTubelegth:
                    break
                end_id = min((segId+1)*(seg_length), num_tube_ftr)
                tmp_rgb_tube_embedding[segId, :] = np.mean(rgb_tube_ftr[start_id:end_id], axis=0)
                tmp_flow_tube_embedding[segId, :] = np.mean(flow_tube_ftr[start_id:end_id], axis=0)
                 
            rgb_tube_embedding[tube_id, ...] = tmp_rgb_tube_embedding
            flow_tube_embedding[tube_id, ...] = tmp_flow_tube_embedding
       
        #pdb.set_trace()
        tube_embedding = np.concatenate((rgb_tube_embedding, flow_tube_embedding), axis=2)
        self.online_cache[i3d_ftr_path] = tube_embedding
        return tube_embedding


    def get_tube_pos_embedding(self, tubeInfo, tube_length, feat_dim=64, feat_type='aiayn'):
        tube_list, frame_list = tubeInfo
        position_mat_raw = torch.zeros((1, self.rpNum, tube_length, 4)) 
        if feat_type=='aiayn':
            bSize = 1
            prpSize = self.rpNum
            kNN = tube_length
            for tubeId, tube in enumerate(tube_list[0]):
                if tubeId>=self.rpNum:
                    break
                tube_length_ori = len(tube)
                tube_seg_length = max(int(tube_length_ori/tube_length), 1)
                
                for tube_seg_id in range(0, tube_length):
                    tube_seg_id_st = tube_seg_id*tube_seg_length
                    tube_seg_id_end = min((tube_seg_id+1)*tube_seg_length, tube_length_ori)
                    if(tube_seg_id_st)>=tube_length_ori:
                        position_mat_raw[0, tubeId, tube_seg_id, :] = position_mat_raw[0, tubeId, tube_seg_id-1, :]
                        continue
                    bbox_list = tube[tube_seg_id_st:tube_seg_id_end]
                    box_np = np.concatenate(bbox_list, axis=0)
                    box_tf = torch.FloatTensor(box_np).view(-1, 4)
                    position_mat_raw[0, tubeId, tube_seg_id, :]= box_tf.mean(0)
            #pdb.set_trace()
            # transform bbx to format with (x_c, yc, w, h) 
            position_mat_raw_v2 = copy.deepcopy(position_mat_raw)
            position_mat_raw_v2[:, 0] = (position_mat_raw[:, 0] + position_mat_raw[:, 2])/2
            position_mat_raw_v2[:, 1] = (position_mat_raw[:, 1] + position_mat_raw[:, 3])/2
            position_mat_raw_v2[:, 2] = position_mat_raw[:, 2] - position_mat_raw[:, 0]
            position_mat_raw_v2[:, 3] = position_mat_raw[:, 3] - position_mat_raw[:, 1]

            pos_emb = extract_position_embedding(position_mat_raw_v2, feat_dim, wave_length=1000)
            
            return pos_emb.squeeze(0)
        else:
            raise  ValueError('%s is not implemented!' %(feat_type))



    def get_visual_item(self, indexOri):
        index = self.use_key_index[indexOri]
        sumInd = 0
        tube_embedding = None
        cap_embedding = None
        cap_length_list = -1
        tAf = time.time() 
        cap_embedding, cap_length_list = self.get_cap_emb(index, self.capNum)
        tBf = time.time() 
        
        if self.use_mean_cache_flag:
            if self.vis_ftr_type=='rgb' or self.vis_ftr_type=='i3d':
                mean_tube_ftr_path = os.path.join(self.mean_cache_ftr_path, self.set_name,\
                       'mean_feature_'+ self.vis_ftr_type + '.h5')
                f_handle = h5py.File(mean_tube_ftr_path, 'r')
                key_str = str(self.maxTubelegth) + '_' + str(index)
                tube_embedding_np = f_handle[key_str][()]
                tube_embedding = torch.FloatTensor(tube_embedding_np).unsqueeze(dim=1)
            
            elif self.vis_ftr_type=='rgb_i3d':
                tube_embedding_list = list()
                for vis_ftr_type  in ['rgb', 'i3d']:
                    mean_tube_ftr_path = os.path.join(self.mean_cache_ftr_path, self.set_name,\
                        'mean_feature_'+ vis_ftr_type + '.h5')
                    f_handle = h5py.File(mean_tube_ftr_path, 'r')
                    key_str = str(self.maxTubelegth) + '_' + str(index)
                    tube_embedding_np = f_handle[key_str][()]
                    tube_embedding_pyt = torch.FloatTensor(tube_embedding_np).unsqueeze(dim=1)
                    tube_embedding_list.append(tube_embedding_pyt)
                tube_embedding = torch.cat(tube_embedding_list, dim=2)

            vd_name, ins_in_vd = self.vid_parser.get_shot_info_from_index(index)
            tube_embedding = torch.FloatTensor(tube_embedding)
            ins_ann, vd_name = self.vid_parser.get_shot_anno_from_index(index)
            tube_info_path = os.path.join(self.tubePath, self.set_name, self.prp_type, str(index)+'.pd') 
            tubeInfo = pickleload(tube_info_path)
            
            if self.pos_type !='none':
                tp1 = time.time() 
                #pdb.set_trace()
                tube_embedding_pos = self.get_tube_pos_embedding(tubeInfo, tube_length=self.maxTubelegth, \
                        feat_dim=self.pos_emb_dim, feat_type=self.pos_type)
                tp2 = time.time()
                tube_embedding_pos = tube_embedding_pos.mean(1).unsqueeze(dim=1)
                tube_embedding = torch.cat((tube_embedding, tube_embedding_pos), dim=2)
            
            return tube_embedding, cap_embedding, tubeInfo, index, cap_length_list, vd_name

        cache_str_shot_str = str(self.maxTubelegth) +'_' + str(index)
        #pdb.set_trace()
        tube_embedding_list = list()
        #pdb.set_trace()
        if cache_str_shot_str in self.cache_ftr_dict.keys():
            if self.vis_ftr_type=='rgb'or self.vis_ftr_type=='rgb_i3d':
                tube_embedding, tubeInfo, tube_to_prp_idx = self.cache_ftr_dict[cache_str_shot_str]
                #print('using cache ftr %s' %(cache_str_shot_str))
            elif self.vis_ftr_type=='i3d':
                tube_embedding, tubeInfo = self.cache_ftr_dict[cache_str_shot_str]
        # get visual tube embedding
        else:
            if self.vis_ftr_type =='rgb'or self.vis_ftr_type=='rgb_i3d':
                tube_embedding, tubeInfo, tube_to_prp_idx  = self.get_tube_embedding(index, self.maxTubelegth, self.out_cache_folder)
                if self.vis_ftr_type=='rgb_i3d':
                    tube_embedding_list.append(tube_embedding)
                tube_embedding = torch.FloatTensor(tube_embedding)
            if self.vis_ftr_type =='i3d' or self.vis_ftr_type=='rgb_i3d':
                tube_embedding  = self.get_tube_embedding_i3d(index, self.maxTubelegth, self.out_cache_folder)
                ins_ann, vd_name = self.vid_parser.get_shot_anno_from_index(index)
                tube_info_path = os.path.join(self.tubePath, self.set_name, self.prp_type, str(index)+'.pd') 
                tubeInfo = pickleload(tube_info_path)
                if self.vis_ftr_type=='rgb_i3d':
                    tube_embedding_list.append(tube_embedding)
                tube_embedding = torch.FloatTensor(tube_embedding)
            
            if self.vis_ftr_type=='rgb_i3d' and cache_str_shot_str not in self.cache_ftr_dict.keys():
                tube_embedding = np.concatenate(tube_embedding_list, axis=2)
            tube_embedding = torch.FloatTensor(tube_embedding)

            # get position embedding
            if self.pos_type !='none':
                tp1 = time.time() 
                #pdb.set_trace()
                tube_embedding_pos = self.get_tube_pos_embedding(tubeInfo, tube_length=self.maxTubelegth, \
                        feat_dim=self.pos_emb_dim, feat_type=self.pos_type)
                tp2 = time.time()
                tube_embedding = torch.cat((tube_embedding, tube_embedding_pos), dim=2)
            
            if self.cache_flag and self.vis_ftr_type=='rgb':
                self.cache_ftr_dict[cache_str_shot_str] = [tube_embedding, tubeInfo, tube_to_prp_idx]
            elif self.cache_flag and self.vis_ftr_type=='rgb_i3d':
                self.cache_ftr_dict[cache_str_shot_str] = [tube_embedding, tubeInfo, tube_to_prp_idx]
            elif self.cache_flag and self.vis_ftr_type=='i3d':
                self.cache_ftr_dict[cache_str_shot_str] = [tube_embedding, tubeInfo]
            tAf2 = time.time()
            #print('extract pos time %f\n' %(tp2-tp1))
        #print('index: %d, caption: %f, visual: %f\n'%(indexOri,  tBf-tAf, tAf2-tBf))
        vd_name, ins_in_vd = self.vid_parser.get_shot_info_from_index(index)
        return tube_embedding, cap_embedding, tubeInfo, index, cap_length_list, vd_name


    def get_tube_info(self, indexOri):
        index = self.use_key_index[indexOri]
        ins_ann, vd_name = self.vid_parser.get_shot_anno_from_index(index)
        tube_info_path = os.path.join(self.tubePath, self.set_name, self.prp_type, str(index)+'.pd') 
        tubeInfo = pickleload(tube_info_path)
        return tubeInfo, index

    def __getitem__(self, index):
        #index =0
        return self.get_visual_item(index)                
        try:
            return self.get_visual_item(index)                
        except:
            print('----------------------bug on %d----------------------------\n' %(index))

def dis_collate_vid(batch):
    ftr_tube_list = list()
    ftr_cap_list = list()
    tube_info_list = list()
    cap_length_list = list()
    index_list = list()
    vd_name_list = list()
    max_length = 0
    for sample in batch:
        ftr_tube_list.append(sample[0])
        ftr_cap_list.append(torch.FloatTensor(sample[1]))
        tube_info_list.append(sample[2])
        index_list.append(sample[3])
        vd_name_list.append(sample[5])

        for tmp_length in sample[4]:
            if(tmp_length>max_length):
                max_length = tmp_length
            cap_length_list.append(tmp_length)

    capMatrix = torch.stack(ftr_cap_list, 0)
    capMatrix = capMatrix[:, :, :max_length, :]
    return torch.stack(ftr_tube_list, 0), capMatrix, tube_info_list, index_list, cap_length_list, vd_name_list

     
if __name__=='__main__':
    from datasetLoader import build_dataloader 
    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name ='train'
    opt.batchSize = 4
    opt.num_workers = 0
    opt.rpNum =30
    opt.vis_ftr_type = 'i3d'
    data_loader, dataset  = build_dataloader(opt)
    for index, input_data in enumerate(data_loader):
        print index
        pdb.set_trace()


    print('finish!')
