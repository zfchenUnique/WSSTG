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

set_debugger()

class actNetDataloader(data.Dataset):
    def __init__(self, set_name_list, dictFile, tubePath, ftrPath, out_cached_folder):
        self.set_name_list = set_name_list
        self.dict = pickleload(dictFile)
        self.rpNum = 20
        self.maxWordNum =20
        self.maxTubelegth = 20
        self.tube_ftr_dim = 2048
        self.tubePath = tubePath
        self.ftrPath = ftrPath
        self.out_cache_folder = out_cached_folder 

    def image_samper_set_up(self, rpNum=20, capNum=1, maxWordNum=20, usedBadWord=False):
        self.rpNum = rpNum
        self.maxWordNum = maxWordNum
        self.usedBadWord = usedBadWord
        self.capNum = capNum
        self.ptdList = list()
        for set_name in self.set_name_list:
            self.ptdList.append(PTD(set_name)) 

    def __len__(self):
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
        tube_prp_path = os.path.join(self.tubePath, set_name, str(shotInfo[1]) + '.pk')
        img_prp_path = os.path.join(self.ftrPath, 'v_'+ shot.video_id+'_rp.h5')
        img_ftr_path = os.path.join(self.ftrPath, 'v_'+ shot.video_id+'_ftr.h5')

        tubeInfo = pickleload(tube_prp_path)
        img_prp_reader = h5py.File(img_prp_path, 'r')
        img_ftr_reader = h5py.File(img_ftr_path, 'r')
        #assert len(img_prp_reader['bbx']) == len(tubeInfo[1])
        tube_list, frame_list = tubeInfo
        frmNum = len(frame_list)
        seg_length = int(frmNum/maxTubelegth)
        tube_to_prp_idx = list()
        ftr_tube_list = list()
        prp_range_num = len(tube_list[0])
        #pdb.set_trace()
        tmp_cache_tube_feature_path = os.path.join(out_cached_folder, \
                    set_name, str(shotInfo[1]) + '.pk')
        if os.path.isfile(tmp_cache_tube_feature_path):
            tmp_tube_ftr_info = pickleload(tmp_cache_tube_feature_path)
            tube_embedding, tubeInfo, tube_to_prp_idx = tmp_tube_ftr_info
            return tube_embedding, tubeInfo, tube_to_prp_idx
        
        for tubeId, tube in enumerate(tube_list[0]):
            if tubeId>= self.rpNum:
                continue
            tube_prp_map = list()
            # find proposals
            for frmId, bbox in enumerate(tube):
                frmName = frame_list[frmId] 
                tmp_bbx = img_prp_reader['bbx'][frmName][:prp_range_num] # to be modified
                tmp_info = img_prp_reader['imfo'][frmName][()].squeeze()
                tmp_bbx[:, 0] = tmp_bbx[:, 0]/tmp_info[1]
                tmp_bbx[:, 2] = tmp_bbx[:, 2]/tmp_info[1]
                tmp_bbx[:, 1] = tmp_bbx[:, 1]/tmp_info[0]
                tmp_bbx[:, 3] = tmp_bbx[:, 3]/tmp_info[0]
                img_prp_res = tmp_bbx - bbox
                img_prp_res_sum = np.sum(img_prp_res, axis=1)
                for prpId in range(self.rpNum):
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
                end_id = min((segId+1)*(seg_length), frmNum)
                tmp_ftr = np.zeros((1, self.tube_ftr_dim), dtype=np.float32)
                for frmId in range(start_id, end_id):
                    frm_name = frame_list[frmId]
                    tmp_ftr +=img_ftr_reader[frm_name][tube_prp_map[frmId]]
                tmp_tube_embedding[segId, :] = tmp_ftr/(end_id-start_id)
            
            tube_embedding[tubeId, ...] = tmp_tube_embedding
        
        if out_cached_folder !='':
            dir_name = os.path.dirname(tmp_cache_tube_feature_path)
            makedirs_if_missing(dir_name)
            pickledump(tmp_cache_tube_feature_path, [tube_embedding, tubeInfo, tube_to_prp_idx])
        return tube_embedding, tubeInfo, tube_to_prp_idx

    def get_visual_item(self, indexOri):
        #pdb.set_trace()
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
            per_des_list = [d.description for d in person_index.descriptions]
            cap_embedding, cap_length_list = self.get_cap_emb_from_list(per_des_list, self.capNum)
            tAf = time.time() 
            print('caption: %d, %f\n'%(indexOri, tAf-tBf))
            # get visual tube embedding 
            shot_id = person_index.shot.id
            tube_embedding, tubeInfo, tube_to_prp_idx  = self.get_tube_embedding([set_idx, shot_id], self.maxTubelegth, self.out_cache_folder)
            tAf2 = time.time() 
            print('visual: %d, %f\n'%(indexOri, tAf2-tAf))
            break
        return tube_embedding, cap_embedding, tubeInfo, person_index, cap_length_list

    def __getitem__(self, index):
        return self.get_visual_item( index)                

def dis_collate(batch):
    ftr_tube_list = list()
    prp_tube_list = list()
    ftr_cap_list = list()
    person_list = list()
    cap_length_list = list()
    max_length = 0
    for sample in batch:
        ftr_tube_list.append(sample[0])
        prp_tube_list.append(sample[1])
        ftr_cap_list.append(sample[2])
        person_list.append(sample[3])

        for tmp_length in sample[4]:
            if(tmp_length>max_length):
                max_length = tmp_length
    
    capMatrix = torch.stack(ftr_cap_list, 0)
    capMatrix = capMatrix[:, :maxLen, :]
    return torch.stack(ftr_tube_list, 0), capMatrix, prp_tube_list, person_index

def build_dataloader(opt):
    if opt.dbSet=='actNet':
        ftrPath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNetPrpsH5'
        tubePath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNetTubePrp'
        dictFile = '../data/dictForDb_actNet.pd'
        out_cached_folder = '/data1/zfchen/data/actNet/actNetTubeCacheFtr'
        if otp.set_name =='train_val':
            set_name_list =['train', 'val']

        dataset = actNetDataloader(set_name_list, dictFile, tubePath \
                , ftrPath, out_cached_folder)
        capNum = 1
        maxWordNum = 20
        rpNum = 20
        dataset.image_samper_set_up(rpNum= rpNum, capNum = capNum, \
                maxWordNum= maxWordNum, usedBadWord=False)
        
        for i in range(len(dataset)):
            if(i<5500):
                continue
            tBf = time.time() 
            outPut = dataset.__getitem__(i)
            tAf = time.time() 
            print('%d/%d, %f\n'%(i, len(dataset),  tAf-tBf))

    else:
        print('Not implemented for dataset %s\n' %(opt.dbSet))
        return
    
    data_loader = data.DataLoader(dataset,  opt.batchSize, \
            num_workers=opt.num_workers, collate_fn=dis_collate, \
            shuffle=True, pin_memory=True)
    return data_loader, dataset

if __name__=='__main__':
    opt = parse_args()
    opt.dbSet = 'actNet'
    data_loader = build_dataloader(opt)
    
