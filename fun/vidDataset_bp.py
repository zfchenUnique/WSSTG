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

set_debugger()

class vidDataloader(data.Dataset):
    def __init__(self, ann_folder, prp_type, set_name, dictFile, tubePath, ftrPath, out_cached_folder):
        self.rpNum = 30
        self.maxWordNum =20
        self.maxTubelegth = 20
        self.tube_ftr_dim = 2048
        self.set_name = set_name
        self.tubePath = tubePath
        self.ftrPath = ftrPath
        self.prp_type = prp_type
        self.vid_parser = vidInfoParser(set_name, ann_folder) 
        self.use_key_index = self.vid_parser.tube_cap_dict.keys()
        self.use_key_index.sort()
        self.info_lines = self.vid_parser.info_lines
        self.

    def __len__(self):
        return len(self.vid_parser.info_lines)

    def get_tube_image_flow(self, index, seg_length=64):
        set_name = self.set_name
        ins_ann, vd_name = self.vid_parser.get_shot_anno_from_index(index)
        tube_info_path = os.path.join(self.tubePath, set_name, self.prp_type, str(index)+'.pd') 
        tubeInfo = pickleload(tube_info_path)
        tube_list, frame_list = tubeInfo
        frmNum = len(frame_list)
        
        tube_to_prp_idx = list()
        ftr_tube_list = list()
        prp_range_num = len(tube_list[0])

        # cache data for saving IO time
        cache_data_dict ={}

        for tubeId, tube in enumerate(tube_list[0]):
            if tubeId>= self.rpNum:
                continue
            tube_prp_map = list()
            # find proposals

            for frmId, bbox in enumerate(tube):
                frmName = frame_list[frmId] 
                frm_full_path = os.path.join()



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
    
        if self.vis_half_size:
            tube_embedding = tube_embedding.view(self.rpNum, self.maxTubelegth/2, 2, self.tube_ftr_dim)
            tube_embedding = np.mean(tube_embedding, axis=2)

        return tube_embedding, tubeInfo, tube_to_prp_idx

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
        #pdb.set_trace()
        index = self.use_key_index[indexOri]
        sumInd = 0
        tube_embedding = None
        cap_embedding = None
        cap_length_list = -1
        tAf = time.time() 
        cap_embedding, cap_length_list = self.get_cap_emb(index, self.capNum)
        tBf = time.time() 
        # get visual tube embedding 
        tube_embedding, tubeInfo, tube_to_prp_idx  = self.get_tube_embedding(index, self.maxTubelegth, self.out_cache_folder)
        tAf2 = time.time()
        tube_embedding = torch.FloatTensor(tube_embedding)
        #print(self.pos_type)
        if self.pos_type !='none':
            tp1 = time.time() 
            #pdb.set_trace()
            tube_embedding_pos = self.get_tube_pos_embedding(tubeInfo, tube_length=self.maxTubelegth, \
                    feat_dim=self.pos_emb_dim, feat_type=self.pos_type)
            tp2 = time.time()
        
            tube_embedding = torch.cat((tube_embedding, tube_embedding_pos), dim=2)
            #print('extract pos time %f\n' %(tp2-tp1))
        #print('index: %d, caption: %f, visual: %f\n'%(indexOri,  tBf-tAf, tAf2-tBf))
        vd_name, ins_in_vd = self.vid_parser.get_shot_info_from_index(index)
        return tube_embedding, cap_embedding, tubeInfo, index, cap_length_list, vd_name

    def __getitem__(self, index):
        return self.get_visual_item(index)                

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

def build_dataloader(opt):
    #pdb.set_trace()
    if opt.dbSet=='vid':
        ftrPath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/vidPrp/Data/VID'
        tubePath = '/data1/zfchen/data/ILSVRC/tubePrp'
        dictFile = '../data/dictForDb_vid.pd'
        out_cached_folder = '/data1/zfchen/data/vid/vidTubeCacheFtr'
        ann_folder = '/data1/zfchen/data/ILSVRC'
        prp_type = 'coco_30_2'
        set_name = opt.set_name
        dataset = vidDataloader(ann_folder, prp_type, set_name, dictFile, tubePath \
                , ftrPath, out_cached_folder)
        capNum = opt.capNum 
        maxWordNum = opt.maxWordNum
        rpNum = opt.rpNum
        pos_emb_dim = opt.pos_emb_dim
        pos_type = opt.pos_type
        #pdb.set_trace()
        dataset.image_samper_set_up(rpNum= rpNum, capNum = capNum, \
                maxWordNum= maxWordNum, usedBadWord=False, \
                pos_emb_dim=pos_emb_dim, pos_type=pos_type)
    else:
        print('Not implemented for dataset %s\n' %(opt.dbSet))
        return
    
    data_loader = data.DataLoader(dataset,  opt.batchSize, \
            num_workers=opt.num_workers, collate_fn=dis_collate_vid, \
            shuffle=True, pin_memory=True)
    return data_loader, dataset
     
if __name__=='__main__':
    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name ='train'
    opt.batchSize = 4
    opt.num_workers = 0
    opt.rpNum =30
    data_loader, dataset  = build_dataloader(opt)
    for index, input_data in enumerate(data_loader):
        print index
    print('finish!')
