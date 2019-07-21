import os
import sys
sys.path.append('..')
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
from vidDataset import *
import h5py

def build_dataloader(opt):
    if opt.dbSet=='vid':
        ftrPath = '' # Path for frame level feature 
        tubePath = '../data/ILSVRC/tubePrp' # Path for information of each tube proposals
        dictFile = '../data/dictForDb_vid_v2.pd' # Path for word embedding 
        out_cached_folder = '../data/vid/vidTubeCacheFtr' # Path for RGB features of tubes
        ann_folder = '../data/ILSVRC' # Path for tube annotations 
        i3d_ftr_path ='../data/video_feature/feature_extraction/tmp/vid' # Path for I3D features
        prp_type = 'coco_30_2' # frame-level proposal extractors
        mean_cache_ftr_path = '../data/vid/meanFeature' 
        ftr_context_path = '../data/vid/coco32/context/Data/VID'

        set_name = opt.set_name
        dataset = vidDataloader(ann_folder, prp_type, set_name, dictFile, tubePath \
                , ftrPath, out_cached_folder)
        capNum = opt.capNum 
        maxWordNum = opt.maxWordNum
        rpNum = opt.rpNum
        pos_emb_dim = opt.pos_emb_dim
        pos_type = opt.pos_type
        vis_ftr_type = opt.vis_ftr_type
        use_mean_cache_flag = opt.use_mean_cache_flag
        context_flag = opt.context_flag
        frm_level_flag = opt.frm_level_flag
        frm_num = opt.frm_num
        dataset.image_samper_set_up(rpNum= rpNum, capNum = capNum, \
                maxWordNum= maxWordNum, usedBadWord=False, \
                pos_emb_dim=pos_emb_dim, pos_type=pos_type, vis_ftr_type=vis_ftr_type, \
                i3d_ftr_path=i3d_ftr_path, use_mean_cache_flag=use_mean_cache_flag,\
                mean_cache_ftr_path=mean_cache_ftr_path, context_flag=context_flag, ftr_context_path=ftr_context_path, frm_level_flag=frm_level_flag, frm_num=frm_num)
        
        shuffle_flag = not opt.no_shuffle_flag 

        data_loader = data.DataLoader(dataset,  opt.batchSize, \
            num_workers=opt.num_workers, collate_fn=dis_collate_vid, \
            shuffle=shuffle_flag, pin_memory=True)
        return data_loader, dataset
    
        data_loader = data.DataLoader(dataset,  opt.batchSize, \
                num_workers=opt.num_workers, collate_fn=dis_collate_actNet, \
                shuffle=shuffle_flag, pin_memory=True)
        return data_loader, dataset

    else:
        print('Not implemented for dataset %s\n' %(opt.dbSet))
        return

if __name__=='__main__':
    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name = 'train'
    opt.vis_ftr_type = 'i3d'
    out_pre = '/data1/zfchen/data/'
    opt.cache_flag = False
    opt.pos_type = 'none'
    data_loader, dataset_ori = build_dataloader(opt)
