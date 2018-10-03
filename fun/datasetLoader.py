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
from actNetDataset import actNetDataloader, dis_collate_actNet

def build_dataloader(opt):
    if opt.dbSet=='vid':
        ftrPath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/vidPrp/Data/VID'
        tubePath = '/data1/zfchen/data/ILSVRC/tubePrp'
        dictFile = '../data/dictForDb_vid.pd'
        out_cached_folder = '/data1/zfchen/data/vid/vidTubeCacheFtr'
        ann_folder = '/data1/zfchen/data/ILSVRC'
        i3d_ftr_path ='/data1/zfchen/code/video_feature/feature_extraction/tmp/vid'
        prp_type = 'coco_30_2'
        if opt.server_id !=36:
            out_cached_folder = '/data1/zfchen/data36/vid/vidTubeCacheFtr'
            ann_folder = '/data1/zfchen/data36/ILSVRC'
            ftrPath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/vidPrp/Data/VID'
            tubePath = '/data1/zfchen/data36/ILSVRC/tubePrp'

        set_name = opt.set_name
        dataset = vidDataloader(ann_folder, prp_type, set_name, dictFile, tubePath \
                , ftrPath, out_cached_folder)
        capNum = opt.capNum 
        maxWordNum = opt.maxWordNum
        rpNum = opt.rpNum
        pos_emb_dim = opt.pos_emb_dim
        pos_type = opt.pos_type
        vis_ftr_type = opt.vis_ftr_type
#        pdb.set_trace()
        dataset.image_samper_set_up(rpNum= rpNum, capNum = capNum, \
                maxWordNum= maxWordNum, usedBadWord=False, \
                pos_emb_dim=pos_emb_dim, pos_type=pos_type, vis_ftr_type=vis_ftr_type, i3d_ftr_path=i3d_ftr_path)
        data_loader = data.DataLoader(dataset,  opt.batchSize, \
            num_workers=opt.num_workers, collate_fn=dis_collate_vid, \
            shuffle=True, pin_memory=True)
        return data_loader, dataset
    
    elif opt.dbSet=='actNet':
        ftrPath = ['/data1/zfchen/data/remote_disk/data10/actNet',
                '/data1/zfchen/data/remote_disk/data7/actNet']
        tubePath = '/data1/zfchen/data/remote_disk/data11/actNet_tube_prp'
        dictFile = '../data/dict_actNet.pd'
        out_cached_folder = '/data1/zfchen/data/actNet/actNetTubeCacheFtr'
        jpg_folder = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNet/actNetPNG'
        i3d_ftr_path = '/data1/zfchen/data/remote_disk/data4/actNet/i3d/actNet/'
        
        if opt.server_id !=36:
            ftrPath = ['/data1/zfchen/data/remote_disk/data10/actNet',
                '/data1/zfchen/data/remote_disk/data7/actNet']
            tubePath = '/data1/zfchen/data/remote_disk/data11/actNet_tube_prp'
            dictFile = '../data/dict_actNet.pd'
            out_cached_folder = '/data1/zfchen/data36/actNet/actNetTubeCacheFtr'
            jpg_folder = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNet/actNetPNG'
            i3d_ftr_path = '/data1/zfchen/data/actNet/i3d/actNet/'
        
        if opt.set_name =='train_val':
            set_name_list = ['train', 'val']
        else:
            set_name_list = [opt.set_name]

        dataset = actNetDataloader(set_name_list, dictFile, tubePath \
                , ftrPath, out_cached_folder)

        dataset.jpg_folder = jpg_folder
        capNum = opt.capNum
        maxWordNum = opt.maxWordNum
        rpNum = opt.rpNum
        vis_ftr_type = opt.vis_ftr_type
        pos_emb_dim = opt.pos_emb_dim
        pos_type = opt.pos_type

        dataset.image_samper_set_up(rpNum= rpNum, capNum= capNum, maxWordNum= maxWordNum, \
                usedBadWord=False, pos_type=pos_type, pos_emb_dim=pos_emb_dim, \
                vis_ftr_type=vis_ftr_type, i3d_ftr_path= i3d_ftr_path)

        data_loader = data.DataLoader(dataset,  opt.batchSize, \
                num_workers=opt.num_workers, collate_fn=dis_collate_actNet, \
                shuffle=True, pin_memory=True)
        return data_loader, dataset

    else:
        print('Not implemented for dataset %s\n' %(opt.dbSet))
        return

if __name__=='__main__':
    opt = parse_args()
    opt.dbSet = 'actNet'
    opt.set_name = 'test'
    data_loader, dataset_ori = build_dataloader(opt)
    for i, input_data in enumerate(data_loader):
        print('%d, batch size: %d\n' %(i, opt.batchSize))

