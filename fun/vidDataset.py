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
from vidDatasetParser import *
from multiprocessing import Process, Pipe, cpu_count, Queue
from vidDatasetParser import vidInfoParser

#set_debugger()

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

    def image_samper_set_up(self, rpNum=20, capNum=1, maxWordNum=20, usedBadWord=False):
        self.rpNum = rpNum
        self.maxWordNum = maxWordNum
        self.usedBadWord = usedBadWord
        self.capNum = capNum

    def __len__(self):
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
            tmp_tube_ftr_info = pickleload(tmp_cache_tube_feature_path)
            tube_embedding, tubeInfo, tube_to_prp_idx = tmp_tube_ftr_info
            if((tube_to_prp_idx[0])>maxTubelegth):
                return tube_embedding, tubeInfo, tube_to_prp_idx
       
        for tubeId, tube in enumerate(tube_list[0]):
            if tubeId>= self.rpNum:
                continue
            tube_prp_map = list()
            # find proposals
            for frmId, bbox in enumerate(tube):
                frmName = frame_list[frmId] 
                img_prp_ftr_info_path = os.path.join(self.ftrPath, self.set_name, vd_name, frmName+ '.pd')
                img_prp_ftr_info = pickleload(img_prp_ftr_info_path) 
                tmp_bbx = img_prp_ftr_info['rois'][:prp_range_num] # to be modified
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
            assert(len(tube_prp_map)==len(tube))
            tube_to_prp_idx.append(tube_prp_map)
            
            # extract features
            tmp_tube_embedding = np.zeros((maxTubelegth, self.tube_ftr_dim), dtype=np.float32)
            for segId in range(maxTubelegth):
                start_id = segId*seg_length
                end_id = (segId+1)*seg_length
                if end_id>frmNum:
                    break
                end_id = min((segId+1)*(seg_length), frmNum)
                tmp_ftr = np.zeros((1, self.tube_ftr_dim), dtype=np.float32)
                for frmId in range(start_id, end_id):
                    frm_name = frame_list[frmId]
                    img_prp_ftr_info_path = os.path.join(self.ftrPath, self.set_name, vd_name, frm_name+ '.pd')
                    img_prp_ftr_info = pickleload(img_prp_ftr_info_path) 
                    tmp_ftr +=img_prp_ftr_info['roiFtr'][tube_prp_map[frmId]]
                tmp_tube_embedding[segId, :] = tmp_ftr/(end_id-start_id)
            
            tube_embedding[tubeId, ...] = tmp_tube_embedding
        
        if out_cached_folder !='':
            dir_name = os.path.dirname(tmp_cache_tube_feature_path)
            makedirs_if_missing(dir_name)
            pickledump(tmp_cache_tube_feature_path, [tube_embedding, tubeInfo, tube_to_prp_idx])
        return tube_embedding, tubeInfo, tube_to_prp_idx

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
        ftr_tube_list.append(torch.FloatTensor(sample[0]))
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
        dataset.image_samper_set_up(rpNum= rpNum, capNum = capNum, \
                maxWordNum= maxWordNum, usedBadWord=False)
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
    opt.batchSize = 1
    opt.num_workers = 8
    data_loader, dataset  = build_dataloader(opt)
    for index, input_data in enumerate(data_loader):
        print index
    print('finish!')
