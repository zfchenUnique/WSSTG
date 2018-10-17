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
import h5py

def pre_load_data(dataset):
    for i in range(len(dataset)):
        t1 = time.time()             
        output= dataset.__getitem__(i)
        t2 = time.time()     
        print('extracting %d: time:%f\n' %(i, t2-t1))

def pre_save_mean_data(dataset, out_dict_name):
    makedirs_if_missing(os.path.dirname(out_dict_name)) 
    file_handle = h5py.File(out_dict_name, 'w')
    #pdb.set_trace()
    for i in range(len(dataset)):
        t1 = time.time()             
        input_data = dataset.__getitem__(i)
        tube_embedding, cap_embedding, tubeInfo, index, cap_length_list, vd_name = input_data

        if not isinstance(index, int): # for actNet dataset
            index = index.shot.id

        cache_str_shot_str = str(dataset.maxTubelegth) +'_' + str(index)
        if cache_str_shot_str not in file_handle.keys():
            file_handle[cache_str_shot_str] = tube_embedding.mean(1).data 
        t2 = time.time()     
        print('extracting %d\%d: time:%f\n' %(i, len(dataset), t2-t1))
    file_handle.close()


def build_dataloader(opt):
    if opt.dbSet=='vid':
        ftrPath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/vidPrp/Data/VID'
        tubePath = '/data1/zfchen/data/ILSVRC/tubePrp'
        dictFile = '../data/dictForDb_vid_v2.pd'
        out_cached_folder = '/data1/zfchen/data/vid/vidTubeCacheFtr'
        ann_folder = '/data1/zfchen/data/ILSVRC'
        i3d_ftr_path ='/data1/zfchen/code/video_feature/feature_extraction/tmp/vid'
        prp_type = 'coco_30_2'
        mean_cache_ftr_path = '/data1/zfchen/data/vid/meanFeature'
        if opt.server_id !=36:
            out_cached_folder = '/data1/zfchen/data36/vid/vidTubeCacheFtr'
            ann_folder = '/data1/zfchen/data36/ILSVRC'
            ftrPath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/vidPrp/Data/VID'
            tubePath = '/data1/zfchen/data36/ILSVRC/tubePrp'
            mean_cache_ftr_path = '/data1/zfchen/data36/vid/meanFeature'

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
#        pdb.set_trace()
        dataset.image_samper_set_up(rpNum= rpNum, capNum = capNum, \
                maxWordNum= maxWordNum, usedBadWord=False, \
                pos_emb_dim=pos_emb_dim, pos_type=pos_type, vis_ftr_type=vis_ftr_type, \
                i3d_ftr_path=i3d_ftr_path, use_mean_cache_flag=use_mean_cache_flag,\
                mean_cache_ftr_path=mean_cache_ftr_path)
        
        if opt.cache_flag == True:
            pre_load_data(dataset)

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
        mean_cache_ftr_path = '/data1/zfchen/data/actNet/meanFeature'
        
        if opt.server_id !=36:
            ftrPath = ['/data1/zfchen/data/remote_disk/data10/actNet',
                '/data1/zfchen/data/remote_disk/data7/actNet']
            #tubePath = '/data1/zfchen/data/remote_disk/data11/actNet_tube_prp'
            tubePath = '/data1/zfchen/data/actNet/actNet_tube_prp'
            dictFile = '../data/dict_actNet.pd'
            out_cached_folder = '/data1/zfchen/data36/actNet/actNetTubeCacheFtr'
            jpg_folder = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNet/actNetPNG'
            i3d_ftr_path = '/data1/zfchen/data/actNet/i3d/actNet/'
            mean_cache_ftr_path = '/data1/zfchen/data/actNet/meanFeature'
        
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
        use_mean_cache_flag = opt.use_mean_cache_flag

        dataset.image_samper_set_up(rpNum= rpNum, capNum= capNum, maxWordNum= maxWordNum, \
                usedBadWord=False, pos_type=pos_type, pos_emb_dim=pos_emb_dim, \
                vis_ftr_type=vis_ftr_type, i3d_ftr_path= i3d_ftr_path, \
                use_mean_cache_flag=use_mean_cache_flag, mean_cache_ftr_path=mean_cache_ftr_path)
        
        if opt.cache_flag == True:
            pre_load_data(dataset)

        data_loader = data.DataLoader(dataset,  opt.batchSize, \
                num_workers=opt.num_workers, collate_fn=dis_collate_actNet, \
                shuffle=True, pin_memory=True)
        return data_loader, dataset

    else:
        print('Not implemented for dataset %s\n' %(opt.dbSet))
        return

if __name__=='__main__':
    opt = parse_args()
    #opt.dbSet = 'actNet'
    opt.dbSet = 'vid'
    opt.set_name = 'train'
    opt.vis_ftr_type = 'i3d'
    out_pre = '/data1/zfchen/data/'
    opt.cache_flag = False
    opt.pos_type = 'none'
    data_loader, dataset_ori = build_dataloader(opt)
    out_dict_name = os.path.join(out_pre, opt.dbSet, 'meanFeature', opt.set_name, 'mean_feature_'+ opt.vis_ftr_type+'.h5')
    pre_save_mean_data(dataset_ori, out_dict_name)
    
    for i in range(len(dataset_ori)):
        print('%d \ %d\n' %(i, len(dataset_ori)))
        index = dataset_ori.use_key_index[i]
        cap_embedding, cap_length_list = dataset_ori.get_cap_emb(index, dataset_ori.capNum)
        if cap_length_list[0]<=0:
            print('%d \%d\n' %(i, len(dataset_ori)))
            pdb.set_trace()
    print('finish')
