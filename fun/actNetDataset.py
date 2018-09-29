import os
import sys
sys.path.append('..')
sys.path.append('../annotations')
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
from actNetDatasetParser import *
from multiGraphAttention import extract_position_embedding 


class actNetDataloader(data.Dataset):
    def __init__(self, set_name_list, dictFile, tubePath, ftrPath, out_cached_folder):
        self.set_name_list = set_name_list
        self.dict = pickleload(dictFile)
        self.rpNum = 20
        self.maxWordNum =20
        self.maxTubelegth = 20
        self.tube_ftr_dim = 2048
        self.prp_type = 'coco_30_2'
        self.tubePath = tubePath
        self.ftrPath = ftrPath
        self.out_cache_folder = out_cached_folder 
        self.cache_ftr_dict = {}
        self.online_cache = {}
        self.cache_ftr_dict_flag = True

    def image_samper_set_up(self, rpNum=20, capNum=1, maxWordNum=20, usedBadWord=False, pos_type='none', pos_emb_dim=64, vis_ftr_type='rgb', i3d_ftr_path=''):
        self.rpNum = rpNum
        self.maxWordNum = maxWordNum
        self.usedBadWord = usedBadWord
        self.capNum = capNum
        self.ptdList = list()
        for set_name in self.set_name_list:
            self.ptdList.append(PTD(set_name)) 
        self.vis_half_size = False
        self.vis_ftr_type = vis_ftr_type
        self.i3d_ftr_path = i3d_ftr_path
        self.pos_emb_dim = pos_emb_dim
        self.pos_type = pos_type
        if self.vis_ftr_type=='i3d':
            self.tube_ftr_dim =1024 # 1024 for rgb, 1024 for flow

    def __len__(self):

        #return 64 # for debuging
        lgh = 0
        for ptd in self.ptdList:
            lgh += len(ptd.people)
        return lgh

    def get_word_emb_from_str(self, capString, maxWordNum):
        capList = capiton_to_word_list(capString) 
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

    def get_cap_emb_from_list(self, per_des_list, capNum):
        # get index
        capIdxList = list()
        if capNum<=len(per_des_list):
            capIdxList = random.sample(range(len(per_des_list)), capNum)
        else:
            for i in range(capNum):
                ele  = random.randint(0, len(det_list)-1)
                capIdxList.append(ele)
        
        # get word embedding
        wordEmbMatrix= np.zeros((capNum, self.maxWordNum, 300), dtype=np.float32)
        cap_length_list = list()
        for i, capIdx in enumerate(capIdxList):
            capString = per_des_list[capIdx]
            wordEmbMatrix[i, ...], valid_length = self.get_word_emb_from_str(capString, self.maxWordNum)
            cap_length_list.append(valid_length)
        return wordEmbMatrix, cap_length_list


    def get_tube_embedding(self, shotInfo, maxTubelegth, out_cached_folder = ''):
        tube_embedding = np.zeros((self.rpNum, maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
        set_name = self.set_name_list[shotInfo[0]]
        shot = self.ptdList[shotInfo[0]].shot(shotInfo[1])
        tube_prp_path = os.path.join(self.tubePath, set_name, self.prp_type, str(shotInfo[1])+'.pd')
        #pdb.set_trace()
        tubeInfo = pickleload(tube_prp_path)

        vd_name = shot.video_id
        
        tube_list, frame_list = tubeInfo
        frmNum = len(frame_list)
        seg_length = int(frmNum/maxTubelegth)
        tube_to_prp_idx = list()
        ftr_tube_list = list()
        prp_range_num = len(tube_list[0])
        #pdb.set_trace()
        tmp_cache_tube_feature_path = os.path.join(out_cached_folder, \
                    set_name, str(shotInfo[1]) + '_'+ str(maxTubelegth) + '.pk')
        if os.path.isfile(tmp_cache_tube_feature_path):
            tmp_tube_ftr_info = pickleload(tmp_cache_tube_feature_path)
            tube_embedding, tubeInfo, tube_to_prp_idx = tmp_tube_ftr_info
            if tube_embedding.shape[0]>=self.rpNum:
                return tube_embedding[:self.rpNum], tubeInfo, tube_to_prp_idx
            else:
                tube_embedding = np.zeros((self.rpNum, maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
        cache_data_dict = {}
        print('set: %s, shot: %d\n' %(set_name, shot.id))

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
                    for tmp_ftr_path  in self.ftrPath:
                        img_prp_ftr_info_path = os.path.join(tmp_ftr_path, 'v_'+ vd_name, frmName+ '.pd')
                        if os.path.isfile(img_prp_ftr_info_path):
                            break
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
                        for tmp_ftr_path  in self.ftrPath:
                            img_prp_ftr_info_path = os.path.join(tmp_ftr_path, 'v_' + vd_name, frmName+ '.pd')
                            if os.path.isfile(img_prp_ftr_info_path):
                                break
                        img_prp_ftr_info = pickleload(img_prp_ftr_info_path) 
                        cache_data_dict[frm_name] = img_prp_ftr_info
                    tmp_ftr +=img_prp_ftr_info['roiFtr'][tube_prp_map[frmId]]
                tmp_tube_embedding[segId, :] = tmp_ftr/(end_id-start_id)
            
            tube_embedding[tubeId, ...] = tmp_tube_embedding
        
        if out_cached_folder !='':
            dir_name = os.path.dirname(tmp_cache_tube_feature_path)
            makedirs_if_missing(dir_name)
            if not os.path.isfile(tmp_cache_tube_feature_path):
                pickledump(tmp_cache_tube_feature_path, [tube_embedding, tubeInfo, tube_to_prp_idx])
    
        if self.vis_half_size:
            tube_embedding = tube_embedding.view(self.rpNum, self.maxTubelegth/2, 2, self.tube_ftr_dim)
            tube_embedding = np.mean(tube_embedding, axis=2)

        return tube_embedding, tubeInfo, tube_to_prp_idx


    def get_tube_embedding_i3d(self, shot_info, maxTubelegth, out_cached_folder = ''):
        #pdb.set_trace()
        index, set_idx = shot_info
        rgb_tube_embedding = np.zeros((self.rpNum, maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
        flow_tube_embedding = np.zeros((self.rpNum, maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
        set_name = self.set_name_list[set_idx]
        i3d_ftr_path =  os.path.join(self.i3d_ftr_path, set_name, str(index) +'.h5')
        if i3d_ftr_path in self.online_cache.keys():
            tube_embedding = self.online_cache[i3d_ftr_path]
            return tube_embedding
        try:
            h5_handle = h5py.File(i3d_ftr_path, 'r')
            for tube_id in range(self.rpNum):
                rgb_tube_ftr = h5_handle[str(tube_id)]['rgb_feature'][()].squeeze()
                flow_tube_ftr = h5_handle[str(tube_id)]['flow_feature'][()].squeeze()
                num_tube_ftr = h5_handle[str(tube_id)]['num_feature'][()].squeeze()
                seg_length = max(int(round(num_tube_ftr/maxTubelegth)), 1)
                tmp_rgb_tube_embedding = np.zeros((maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
                tmp_flow_tube_embedding = np.zeros((maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
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
        except:
            print('fail to open %s\n' %(i3d_ftr_path))
            tmp_error_fn = './fail_i3d_%s.txt' %(set_name)
            f_handle = open(tmp_error_fn, 'a')
            f_handle.write('%s, %d\n' %(set_name, index))
        #pdb.set_trace()
        tube_embedding = np.concatenate((rgb_tube_embedding, flow_tube_embedding), axis=2)
        #self.online_cache[i3d_ftr_path] = tube_embedding
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


    def get_tube_info(self, index):
        for set_idx, ptd in enumerate(self.ptdList):
            lghSet = len(ptd.people)
            if index>=lghSet:
                index -=lghSet
                continue
            set_name = self.set_name_list[set_idx]
            person_index = ptd.person(index+1)
            shot_id = person_index.shot.id
            tube_prp_path = os.path.join(self.tubePath, set_name, self.prp_type, str(shot_id)+'.pd')            
            tubeInfo = pickleload(tube_prp_path)
        return tubeInfo, person_index

    def get_visual_item(self, indexOri):
        index = indexOri
        sumInd = 0
        tube_embedding = None
        cap_embedding = None
        person_index = None
        cap_length_list = -1
        for set_idx, ptd in enumerate(self.ptdList):
            lghSet = len(ptd.people)
            if index>=lghSet:
                index -=lghSet
                continue
            
            tBf = time.time() 
            person_index = ptd.person(index+1)
            per_des_list = [d.description.encode('utf-8') for d in person_index.descriptions]
            cap_embedding, cap_length_list = self.get_cap_emb_from_list(per_des_list, self.capNum)
            tAf = time.time() 
            #print('caption: %d, %f\n'%(indexOri, tAf-tBf))
            # get visual tube embedding 
            shot_id = person_index.shot.id
            cache_str_shot_str = str(set_idx) +'_' + str(shot_id)
            if cache_str_shot_str in self.cache_ftr_dict.keys():
                if self.vis_ftr_type=='rgb':
                    tube_embedding, tubeInfo, tube_to_prp_idx = self.cache_ftr_dict[cache_str_shot_str]
                    print('using cache ftr %s' %(cache_str_shot_str))
                elif self.vis_ftr_type=='i3d':
                    tube_embedding, tubeInfo = self.cache_ftr_dict[cache_str_shot_str]
            else:
                if self.vis_ftr_type == 'rgb':
                    tube_embedding, tubeInfo, tube_to_prp_idx  = self.get_tube_embedding([set_idx, shot_id], self.maxTubelegth, self.out_cache_folder)
                elif self.vis_ftr_type =='i3d':
                    tube_embedding  = self.get_tube_embedding_i3d([shot_id, set_idx], self.maxTubelegth, self.out_cache_folder)
                    set_name = self.set_name_list[set_idx]
                    tube_prp_path = os.path.join(self.tubePath, set_name, self.prp_type, str(shot_id)+'.pd') 
                    tubeInfo = pickleload(tube_prp_path)
                    
            tAf2 = time.time() 
            tube_embedding = torch.FloatTensor(tube_embedding)
            #print('visual: %d, %f\n'%(indexOri, tAf2-tAf))
            if self.pos_type !='none':
                tp1 = time.time() 
            #pdb.set_trace()
                tube_embedding_pos = self.get_tube_pos_embedding(tubeInfo, tube_length=self.maxTubelegth, \
                    feat_dim=self.pos_emb_dim, feat_type=self.pos_type)
                tp2 = time.time()
                tube_embedding = torch.cat((tube_embedding, tube_embedding_pos), dim=2)

            if self.cache_ftr_dict_flag and self.vis_ftr_type=='rgb':
                self.cache_ftr_dict[cache_str_shot_str] = [tube_embedding, tubeInfo, tube_to_prp_idx]
            elif self.cache_ftr_dict_flag and self.vis_ftr_type=='i3d':
                self.cache_ftr_dict[cache_str_shot_str] = [tube_embedding, tubeInfo]
            break
        return tube_embedding, cap_embedding, tubeInfo, person_index, cap_length_list, shot_id

    def __getitem__(self, index):
        return self.get_visual_item( index)                

def dis_collate_actNet(batch):
    ftr_tube_list = list()
    prp_tube_list = list()
    ftr_cap_list = list()
    person_list = list()
    cap_length_list = list()
    shot_list = list()
    max_length = 0
    for sample in batch:
        ftr_tube_list.append(sample[0])
        ftr_cap_list.append(torch.FloatTensor(sample[1]))
        prp_tube_list.append(sample[2])
        person_list.append(sample[3])
        shot_list.append(sample[5])

        for tmp_length in sample[4]:
            if(tmp_length>max_length):
                max_length = tmp_length
            cap_length_list.append(tmp_length)
    
    capMatrix = torch.stack(ftr_cap_list, 0)
    capMatrix = capMatrix[:, :max_length, :]
    return torch.stack(ftr_tube_list, 0), capMatrix, prp_tube_list, person_list, cap_length_list, shot_list

if __name__=='__main__':
    opt = parse_args()
    opt.dbSet = 'actNet'
    opt.num_workers = 0
    data_loader, dataset_ori = build_dataloader(opt)
    for i, input_data in enumerate(data_loader):
        print(i)
    
