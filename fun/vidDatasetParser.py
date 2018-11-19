
import sys
sys.path.append('../')
sys.path.append('../util')
from util.mytoolbox import *
import pdb
import h5py
import csv
import numpy as np
from netUtil import *
sys.path.append('../annotations')
#from pathos.multiprocessing import ProcessingPool as Pool

from multiprocessing import Process, Pipe, cpu_count, Queue
from itertools import izip
import multiprocessing
from multiprocessing import Pool
import dill
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wsParamParser import parse_args
import random
import operator
from nltk.corpus import stopwords
set_debugger()
from fun.datasetLoader import *
import math

def spawn(f):
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p,c) in pipe]


def apply_func(f, q_in, q_out):
    while not q_in.empty():
        i, x = q_in.get()
        q_out.put((i, f(x)))

# map a function using a pool of processes
def parmapV2(f, X, nprocs = cpu_count()):
    q_in, q_out   = Queue(), Queue()
    proc = [Process(target=apply_func, args=(f, q_in, q_out)) for _ in range(nprocs)]
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [p.start() for p in proc]
    res = [q_out.get() for _ in sent]
    [p.join() for p in proc]

    return [x for i,x in sorted(res)]

class vidInfoParser(object):
    def __init__(self, set_name, annFd):
        self.tube_gt_path = os.path.join(annFd, 'Annotations/VID/tubeGt', set_name)
        self.tube_name_list_fn = os.path.join(annFd, 'Data/VID/annSamples/', set_name+'_valid_list.txt')
        self.jpg_folder = os.path.join(annFd, 'Data/VID/', set_name)
        self.info_lines = textread(self.tube_name_list_fn)
        self.set_name = set_name
        self.tube_ann_list_fn = os.path.join(annFd, 'Data/VID/annSamples/', set_name + '_ann_list_v2.txt')
        #pdb.set_trace()
        ins_lines = textread(self.tube_ann_list_fn)
        ann_dict_set_dict = {}
        for line in ins_lines:
            ins_id_str, caption = line.split(',', 1)
            ins_id = int(ins_id_str)
            if ins_id not in ann_dict_set_dict.keys():
                ann_dict_set_dict[ins_id] = list()
            ann_dict_set_dict[ins_id].append(caption)
        self.tube_cap_dict = ann_dict_set_dict

    def get_length(self):
        return len(self.info_lines)

    def get_shot_info_from_index(self, index):
        info_Str = self.info_lines[index]
        vd_name, ins_id_str = info_Str.split(',')
        return vd_name,  ins_id_str

    def get_shot_anno_from_index(self, index):
        vd_name, ins_id_str =  self.get_shot_info_from_index(index)
        jsFn = os.path.join(self.tube_gt_path, vd_name + '.js')
        annDict = jsonload(jsFn)
        ann = None
        for ii, ann in enumerate(annDict['annotations']):
            track = ann['track']
            trackId = ann['id']
            if(trackId!=ins_id_str):
                continue
            break;
        return ann, vd_name

    def get_shot_frame_list_from_index(self, index):
        ann, vd_name = self.get_shot_anno_from_index(index)
        frm_list = list()
        track = ann['track']
        trackId = ann['id']
        frmNum = len(track)
        for iii in range(frmNum):
            vdFrmInfo = track[iii]
            imPath = '%06d' %(vdFrmInfo['frame']-1)
            frm_list.append(imPath)
        return frm_list, vd_name

    def proposal_path_set_up(self, prpPath):
        self.propsal_path = os.path.join(prpPath, self.set_name)


def get_all_instance_frames(set_name, annFd):
    tubeGtPath = os.path.join(annFd, 'Annotations/VID/tubeGt', set_name)
    tube_name_list_fn = os.path.join(annFd, 'Data/VID/annSamples/', set_name+'_valid_list.txt')
    jpg_folder = os.path.join(annFd, 'Data/VID/', set_name)
    all_frm_path_list = list()

    info_lines = textread(tube_name_list_fn)
    for i, vd_info  in enumerate(info_lines):
        #if(i!=300):
        #    continue
        vd_name, ins_id_str = vd_info.split(',')
        jsFn = os.path.join(tubeGtPath, vd_name + '.js')
        annDict = jsonload(jsFn)
        for ii, ann in enumerate(annDict['annotations']):
            track = ann['track']
            trackId = ann['id']
            if(trackId!=ins_id_str):
                continue
            frmNum = len(track)
            for iii in range(frmNum):
                vdFrmInfo = track[iii]
                imPath = jpg_folder + '/' + vd_name + '/' + '%06d.JPEG' %(vdFrmInfo['frame']-1)
                all_frm_path_list.append(imPath)
            break;
    all_frm_path_list_unique = list(set(all_frm_path_list))
    all_frm_path_list_unique.sort()
    return all_frm_path_list_unique

def extract_shot_prp_list_from_pickle(vidParser, shot_index, prp_num=20,do_norm=1):
    frm_list, vd_name = vidParser.get_shot_frame_list_from_index(shot_index)
    prp_list = list()

    for i, frm_name in enumerate(frm_list):
        frm_name_raw = frm_name.split('.')[0]
        prp_path = os.path.join(vidParser.propsal_path, vd_name, frm_name_raw+'.pd')
        frm_prp_info = cPickleload(prp_path)
        tmp_bbx = frm_prp_info['rois'][:prp_num]
        tmp_score = frm_prp_info['roisS'][:prp_num]
        tmp_info = frm_prp_info['imFo'].squeeze()
        
        #pdb.set_trace() 
        if do_norm==1:
            tmp_bbx[:, 0] = tmp_bbx[:, 0]/tmp_info[1]
            tmp_bbx[:, 2] = tmp_bbx[:, 2]/tmp_info[1]
            tmp_bbx[:, 1] = tmp_bbx[:, 1]/tmp_info[0]
            tmp_bbx[:, 3] = tmp_bbx[:, 3]/tmp_info[0]
        elif do_norm==2:
            tmp_bbx[:, 0] = tmp_bbx[:, 0]*tmp_info[2]/tmp_info[1]
            tmp_bbx[:, 2] = tmp_bbx[:, 2]*tmp_info[2]/tmp_info[1]
            tmp_bbx[:, 1] = tmp_bbx[:, 1]*tmp_info[2]/tmp_info[0]
            tmp_bbx[:, 3] = tmp_bbx[:, 3]*tmp_info[2]/tmp_info[0]
        else:
            tmp_bbx = tmp_bbx/tmp_info[2]
        tmp_score = np.expand_dims(tmp_score, axis=1 )
        prp_list.append([tmp_score, tmp_bbx])
    return prp_list, frm_list, vd_name

def resize_tube_bbx(tube_vis, frmImList_vis):
    for prpId, frm in enumerate(tube_vis):
        h, w, c = frmImList_vis[prpId].shape
        tube_vis[prpId][0] = tube_vis[prpId][0]*w
        tube_vis[prpId][2] = tube_vis[prpId][2]*w
        tube_vis[prpId][1] = tube_vis[prpId][1]*h
        tube_vis[prpId][3] = tube_vis[prpId][3]*h
    return tube_vis

def evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, thre=0.5 ,topKOri=20, more_detailed_flag=False):
    #pdb.set_trace()
    topK = min(topKOri, len(shot_proposals[0][0]))
    recall_k = [0.0] * (topK + 1)
    iou_list = list()
    ann, vid_name = vid_parser.get_shot_anno_from_index(tube_index) 
    boxes = {}
    for i, ann_frame in enumerate(ann['track']):
        frame_ind = ann_frame['frame']
        box = ann_frame['bbox']
        h, w = ann_frame['frame_size']
        box[0] = box[0]*1.0/w
        box[2] = box[2]*1.0/w
        box[1] = box[1]*1.0/h
        box[3] = box[3]*1.0/h
        keyName = '%06d' %(frame_ind-1)
        boxes[keyName] = box

#    pdb.set_trace()
    tube_list, frame_list = shot_proposals
    assert(len(tube_list[0][0])== len(frame_list))
    is_instance_annotated = False
    for i in range(topK):
        recall_k[i+1] = recall_k[i]
        if is_instance_annotated:
            continue
        curTubeOri = tube_list[0][i]
        tube_key_bbxList = {}
        for frame_ind, gt_box in boxes.iteritems():
            try:
                index_tmp = frame_list.index(frame_ind)
                tube_key_bbxList[frame_ind] = curTubeOri[index_tmp]
            except:
                print('key %s do not exist in shot' %(frame_ind))
        #pdb.set_trace()
        ol = compute_LS(tube_key_bbxList, boxes) 
        if ol < thre:
            if more_detailed_flag:
                iou_list.append(ol)
            continue
        else:
            recall_k[i+1] += 1.0
            is_instance_annotated = True
            if more_detailed_flag:
                iou_list.append(ol)
    if more_detailed_flag:
        return recall_k, iou_list
    else:
        return recall_k

def multi_process_connect_tubes(param_list):
    tube_index, tube_save_path, prp_num, tube_model_name, connect_w, set_name, annFd, vid_parser = param_list
    #vid_parser = vidInfoParser(set_name, annFd)
    print(tube_save_path)
    if os.path.isfile(tube_save_path):
        print('\n file exist\n')
        return
    if tube_model_name =='coco' :
        prp_list, frmList, vd_name = extract_shot_prp_list_from_pickle(vid_parser, tube_index, prp_num, do_norm=1)
    else:
        prp_list, frmList, vd_name = extract_shot_prp_list_from_pickle(vid_parser, tube_index, prp_num, do_norm=2)
    results = get_tubes(prp_list, connect_w)
    shot_proposals = [results, frmList]
    makedirs_if_missing(os.path.dirname(tube_save_path))        
    pickledump(tube_save_path, shot_proposals)


def test_im_prp():
    #set_name = 'val'
    set_name = 'train'
    annFd = '/data1/zfchen/data/ILSVRC'
    #prpFd = '/data1/zfchen/data/ILSVRC/vid_prp_zf/Data/VID'
    #prpFd = '/data1/zfchen/data/ILSVRC/vid_prp_vg/Data/VID'
    prpFd = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/vidPrp/Data/VID'
    #prpFd = '/data1/zfchen/data/ILSVRC/vid_prp/Data/VID'
    tube_model_name = 'coco'
    #tube_model_name = 'vg'
    #tube_model_name = 'zf'
    vis_frame_num = 30
    prp_num = 30
    connect_w =0.2
    tube_index = 0
    thre = 0.5
    cpu_num =20
    vid_parser = vidInfoParser(set_name, annFd)
    vid_parser.proposal_path_set_up(prpFd)
    topK = prp_num
    recall_k2_sum = np.array([0.0] * (topK + 1))
    recall_k3_sum = np.array([0.0] * (topK + 1))
    recall_k4_sum = np.array([0.0] * (topK + 1))
    recall_k5_sum = np.array([0.0] * (topK + 1))
    pool_job_list = list()

    for tube_index in range(0, vid_parser.get_length()):
    #for tube_index in range(300, 301):
        tube_save_path = os.path.join(annFd, 'tubePrp' ,set_name, tube_model_name + '_' + str(prp_num) +'_' + str(int(10*connect_w)) , str(tube_index) + '.pd')
        #multi_process_connect_tubes([tube_index, tube_save_path, prp_num, tube_model_name, connect_w, set_name, annFd, vid_parser])
        #res = apply_async(pool, multi_process_connect_tubes, (tube_index, tube_save_path, prp_num, tube_model_name, connect_w, set_name, annFd, vid_parser))
        #res.get()
        pool_job_list.append((tube_index, tube_save_path, prp_num, tube_model_name, connect_w, set_name, annFd, vid_parser))
    #print(pool_job_list)
    for stIdx in range(0, len(pool_job_list), cpu_num):
        edIdx = stIdx + cpu_num
        if edIdx>len(pool_job_list):
            edIdx = len(pool_job_list)
        parmap(multi_process_connect_tubes, pool_job_list[stIdx: edIdx])
    print('finish getting tubes')
    return
    pdb.set_trace()
        #if os.path.isfile(tube_save_path):
        #    continue
    
    for tube_index in range(0, vid_parser.get_length()):
        tube_save_path = os.path.join(annFd, 'tubePrp' ,set_name, tube_model_name + '_' + str(prp_num) +'_' + str(int(10*connect_w)) , str(tube_index) + '.pd')
        #if os.path.isfile(tube_save_path):
        #    continue
        print(tube_save_path)
        if tube_model_name =='coco' :
            prp_list, frmList, vd_name = extract_shot_prp_list_from_pickle(vid_parser, tube_index, prp_num, do_norm=1)
        else:
            prp_list, frmList, vd_name = extract_shot_prp_list_from_pickle(vid_parser, tube_index, prp_num, do_norm=2)
        results = get_tubes(prp_list, connect_w)
        shot_proposals = [results, frmList]
        makedirs_if_missing(os.path.dirname(tube_save_path))        
        pickledump(tube_save_path, shot_proposals)
        
        recallK2 = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, 0.2 ,topKOri=prp_num)
        recallK3 = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, 0.3 ,topKOri=prp_num)
        recallK4 = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, 0.4 ,topKOri=prp_num)
        recallK5 = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, thre ,topKOri=prp_num)
        recall_k2_sum += np.array(recallK2)
        recall_k3_sum += np.array(recallK3)
        recall_k4_sum += np.array(recallK4)
        recall_k5_sum += np.array(recallK5)
        
        print('%d/%d' %(tube_index, vid_parser.get_length())) 
        print('thre: %f %f %f %f\n' %( 0.2,0.3,0.4,0.5))
        print((recall_k2_sum)*1.0/(tube_index+1))
        print((recall_k3_sum)*1.0/(tube_index+1))
        print((recall_k4_sum)*1.0/(tube_index+1))
        print((recall_k5_sum)*1.0/(tube_index+1))

        #continue
        # visualization
        pdb.set_trace() 
        frmImNameList = [os.path.join(vid_parser.jpg_folder, vd_name, frame_name + '.JPEG') for frame_name in frmList]
        frmImList = list()
        for fId, imPath  in enumerate(frmImNameList):
            img = cv2.imread(imPath)
            frmImList.append(img)
        visIner = int(len(frmImList) /vis_frame_num)
        #pdb.set_trace() 
        for ii in range(len(results[0])):
            print('visualizing tube %d\n'%(ii))
            tube = results[0][ii]
            frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
            tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
            tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
            vd_name_raw = vd_name.split('/')[-1]
            visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis_resize, 'sample/'+vd_name_raw+ '_'+str(prp_num) + str(ii)+'.gif')


def visual_tube_proposals(tube_save_path, vid_parser, tube_index, prp_num):
    
    topK = prp_num
    recall_k2_sum = np.array([0.0] * (topK + 1))
    recall_k3_sum = np.array([0.0] * (topK + 1))
    recall_k4_sum = np.array([0.0] * (topK + 1))
    recall_k5_sum = np.array([0.0] * (topK + 1))
    
    shot_proposals = pickleload(tube_save_path)
    results, frmList = shot_proposals
    vd_name, ins_id_str = vid_parser.get_shot_info_from_index(tube_index)
    
    recallK2 = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, 0.2 ,topKOri=prp_num)
    recallK3 = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, 0.3 ,topKOri=prp_num)
    recallK4 = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, 0.4 ,topKOri=prp_num)
    recallK5 = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, 0.5 ,topKOri=prp_num)
    recall_k2_sum += np.array(recallK2)
    recall_k3_sum += np.array(recallK3)
    recall_k4_sum += np.array(recallK4)
    recall_k5_sum += np.array(recallK5)
    
    print('%d/%d' %(tube_index, vid_parser.get_length())) 
    print('thre: %f %f %f %f\n' %( 0.2,0.3,0.4,0.5))
    print((recall_k2_sum)*1.0/(tube_index+1))
    print((recall_k3_sum)*1.0/(tube_index+1))
    print((recall_k4_sum)*1.0/(tube_index+1))
    print((recall_k5_sum)*1.0/(tube_index+1))

    #continue
    # visualization
    frmImNameList = [os.path.join(vid_parser.jpg_folder, vd_name, frame_name + '.JPEG') for frame_name in frmList]
    frmImList = list()
    for fId, imPath  in enumerate(frmImNameList):
        img = cv2.imread(imPath)
        frmImList.append(img)
    vis_frame_num = 30
    visIner = int(len(frmImList) /vis_frame_num)
    #pdb.set_trace() 
    for ii in range(len(results[0])):
        print('visualizing tube %d\n'%(ii))
        tube = results[0][ii]
        frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
        tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
        tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
        vd_name_raw = vd_name.split('/')[-1]
        visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis_resize, 'sample/'+vd_name_raw+ '_'+str(prp_num) + str(ii)+'.gif')


###############################################################################################
def get_recall_for_tube_proposals(tube_save_path, vid_parser, tube_index, prp_num, thre_list=[0.2, 0.3, 0.4, 0.5]):
    
    topK = prp_num
    
    shot_proposals = pickleload(tube_save_path)
    results, frmList = shot_proposals
    vd_name, ins_id_str = vid_parser.get_shot_info_from_index(tube_index)
   
    recall_list = list()
    for thre in thre_list: 
        recallK = evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, thre ,topKOri=prp_num)
        recall_list.append(np.array(recallK))
    return recall_list

#####################################################################################
def vis_im_prp():
    #set_name = 'val'
    set_name = 'train'
    annFd = '/data1/zfchen/data/ILSVRC'
    #prpFd = '/data1/zfchen/data/ILSVRC/vid_prp_zf/Data/VID'
    prpFd = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/vidPrp/Data/VID'
    #prpFd = '/data1/zfchen/data/ILSVRC/vid_prp_vg/Data/VID'
    tube_model_name = 'coco'
    #tube_model_name = 'vg'
    #tube_model_name = 'zf'
    vis_frame_num = 30
    prp_num = 30
    connect_w =0.2
    tube_index = 0
    thre = 0.5
    vis_flag = True
    vid_parser = vidInfoParser(set_name, annFd)
    vid_parser.proposal_path_set_up(prpFd)

    for tube_index in range(100, vid_parser.get_length()):
    #for tube_index in range(0, 2):
        tube_save_path = os.path.join(annFd, 'tubePrp' ,set_name, tube_model_name + '_' + str(prp_num) +'_' + str(int(10*connect_w)) , str(tube_index) + '.pd')

        if not vis_flag:
            continue
        visual_tube_proposals(tube_save_path, vid_parser, tube_index, prp_num)
        pdb.set_trace()

#####################################################################################
def show_distribute_over_categories(recall_list, ann_list, thre_list):
    #pdb.set_trace()
    print('average instance level performance')
    recall_ins_sum = list()
    for ii, thre in enumerate(thre_list):
        tmp_ins_sum =  np.array([0.0] * (recall_list[0][ii].shape[0]))
        for i, recall_ins in enumerate(recall_list) :
            tmp_ins_sum +=recall_ins[ii]
        recall_ins_sum.append( tmp_ins_sum /len(recall_list))
        print('thre@ins@%f, %f\n' %(thre, recall_ins_sum[ii][-1]))
        print('top K, recall')
        print(recall_ins_sum[ii])
    pdb.set_trace()

    print('showing recall distribution over categories')
    recall_k_categories_dict = {}
    for i, ann in enumerate(ann_list):
        class_id =  str(ann['track'][0]['class'])
        if class_id in recall_k_categories_dict.keys():
            recall_cat_list, ins_cat_num = recall_k_categories_dict[class_id]
            for ii, recall_thre in enumerate(recall_list[i]):
                recall_cat_list[ii] += recall_thre 
            ins_cat_num +=1
            recall_k_categories_dict[class_id] =[recall_cat_list, ins_cat_num]
        else:
            ins_cat_num =1
            recall_k_categories_dict[class_id] = [recall_list[i], ins_cat_num]

    mean_cat_map = list()
    for i, thre in enumerate(thre_list):
        print('--------------------------------------------------------\n')
        print('recall@%f\n' %(thre))
        recall_plot = list()
        for ii, cat_name in enumerate(recall_k_categories_dict.keys()):
            recall_cat_list, ins_num = recall_k_categories_dict[cat_name]
            recall_thre = recall_cat_list[i][-1]*1.0/ins_num
            print('%s: %f\n' %(cat_name, recall_thre))
            recall_plot.append(recall_thre)
        cat_list = recall_k_categories_dict.keys()
        plt.close()
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.barh(range(len(cat_list)), recall_plot, tick_label=cat_list)
        plt.show()
        bar_name = './sample/vid_recall_train@%d.jpg' %(int(thre*10))
        plt.savefig(bar_name)
        mean_cat_map.append(sum(recall_plot)/float(len(recall_plot)))
    for thre, mAp in zip(thre_list, mean_cat_map):
        print('thre@%f , map: %f\n' %(thre, mAp))


def show_distribute_over_object_size(recall_list, ann_list, thre_list):
    print('showing recall distribution over object size')

def statistic_im_prp():
    set_name = 'train'
    #set_name = 'train'
    annFd = '/data1/zfchen/data/ILSVRC'
    #prpFd = '/data1/zfchen/data/ILSVRC/vid_prp_zf/Data/VID'
    prpFd = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/vidPrp/Data/VID'
    #prpFd = '/data1/zfchen/data/ILSVRC/vid_prp/Data/VID'
    tube_model_name = 'coco'
    #tube_model_name = 'vg'
    #tube_model_name = 'zf'
    vis_frame_num = 30
    prp_num = 30
    connect_w =0.2
    tube_index = 0
    thre_list = [0.2, 0.3, 0.4, 0.5]
    vis_flag = False
    vid_parser = vidInfoParser(set_name, annFd)
    vid_parser.proposal_path_set_up(prpFd)
    
    ann_list = list() 
    recall_list = list()
    for tube_index in range(0, vid_parser.get_length()):
    #for tube_index in range(0, 50):
        tube_save_path = os.path.join(annFd, 'tubePrp' ,set_name, tube_model_name + '_' + str(prp_num) +'_' + str(int(10*connect_w)) , str(tube_index) + '.pd')
        ann, vd_str  = vid_parser.get_shot_anno_from_index(tube_index)
        recall_k = get_recall_for_tube_proposals(tube_save_path, vid_parser, tube_index, prp_num)
        ann_list.append(ann)
        recall_list.append(recall_k)
    show_distribute_over_categories(recall_list, ann_list, thre_list)


def vid_split_validation_test():
    #pdb.set_trace()
    # setting up random seed
    seed_value =0
    np.random.seed(seed_value) # cpu vars
    random.seed(seed_value)

    # file path
    val_ann_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_ann_list_v2_ori.txt'
    val_split_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_ann_list_v2.txt'
    test_split_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/test_ann_list_v2.txt'
    val_list = textread(val_ann_list_fn)
    file_num = len(val_list) 
    per_list = np.random.permutation(file_num) 
    
    new_val_list = list()
    new_test_list = list()

    file_num_new_val = int(file_num/2)
    new_val_list = [val_list[per_list[i]] for i in range(file_num_new_val) ]
    new_test_list = [val_list[per_list[i]] for i in range(file_num_new_val, file_num) ]
    
    textdump(val_split_list_fn, new_val_list)
    textdump(test_split_list_fn, new_test_list)

def vid_txt_only():
    train_ann_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/train_ann_list_v2.txt'  
    val_ann_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_ann_list_v2_ori.txt'
    train_ann_list_fn_only = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/train_ann_list_v2_txt_only.txt'  
    val_ann_list_fn_only = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_ann_list_v2_txt_only.txt'

    train_list = textread(train_ann_list_fn)
    val_list = textread(val_ann_list_fn)

    train_list_cap_only = [ txt_ele.split(',', 1)[1] for txt_ele in train_list]
    val_list_cap_only = [ txt_ele.split(',', 1)[1] for txt_ele in val_list]
    
    textdump(train_ann_list_fn_only, train_list_cap_only)
    textdump(val_ann_list_fn_only, val_list_cap_only)


def vid_caption_processing():
    cap_folder = '/data1/zfchen/data/ILSVRC/capResults'
    train_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/train_valid_list.txt'  
    val_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_valid_list.txt'
    ins_cap_list_fn = '/data1/zfchen/data/ILSVRC/capResults/instance_annoatation_list_v2_check.txt'
    train_ann_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/train_ann_list_v2.txt'  
    val_ann_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_ann_list_v2.txt'
    
    train_list = textread(train_list_fn)    
    val_list = textread(val_list_fn)
    ins_ann_list = textread(ins_cap_list_fn)

    len_train_ins = len(train_list)
    train_ins_ann_list = list()
    val_ins_ann_list = list()
    for i, ins_line in enumerate(ins_ann_list):
        val_flag = False
        ins_id_str, caption = ins_line.split(',', 1)
        ins_id = int(ins_id_str)
        if ins_id>=len_train_ins:
            ins_id -=len_train_ins
            val_flag=True
        if not val_flag:
            train_ins_ann_list.append('%d,%s' %(ins_id, caption))
        else:
            val_ins_ann_list.append('%d,%s' %(ins_id, caption))
    textdump(train_ann_list_fn, train_ins_ann_list)
    textdump(val_ann_list_fn, val_ins_ann_list)

def vid_valid_caption_preprocessing():
    pdb.set_trace()
    cap_folder = '/data1/zfchen/data/ILSVRC/capResults'
    min_cap_length = 5
    dir_list = os.listdir(cap_folder)
    ann_dict_set_dict = {}
    for sub_dir in dir_list:
        full_sub_path = os.path.join(cap_folder, sub_dir)
        if os.path.isdir(full_sub_path) == True:
            ann_file_list = os.listdir(full_sub_path)
            ann_check_folder = os.path.join(cap_folder, sub_dir, 'check')
            for i, ann_fn in enumerate(ann_file_list):
                if os.path.isdir(ann_fn):
                    continue
                if ann_fn.split('.')[-1]!='txt':
                    print('invalid file: %s\n' %(ann_fn))
                    continue
                ann_parts = ann_fn.split('_')
                if len(ann_parts) !=5:
                    print('invalid file: %s\n' %(ann_fn))
                    continue
                ann_file_full_path = os.path.join(cap_folder, sub_dir, ann_fn)
                ins_id = int(ann_parts[1])
                lines  = textread(ann_file_full_path)

                cap_str_record = None
                ann_check_fn = os.path.join(ann_check_folder, ann_parts[0]+'_' + ann_parts[1]+'.txt')
                ann_check_list = list()
                if os.path.isfile(ann_check_fn):
                    ann_check_list = textread(ann_check_fn)
               
                if ins_id==75:
                    pdb.set_trace()


                for ii, line_str in enumerate(lines):
                    line_str_part = line_str.split(",", 9) # bug on python 2.x
                    if(len(line_str_part)>10):
                        print(line_str_part)
                        pdb.set_trace()
                    if len(line_str_part)<10:
                        print(line_str_part)
                        pdb.set_trace()
                    cap_str = line_str_part[-1]
                    # update annotation file
                    if len(ann_check_list)>ii:
                        if(ann_check_list[ii]=='0'):
                            continue
                        elif(ann_check_list[ii]=='1'):
                            cap_str_record = cap_str.replace('_', ' ') # replace annos like 'giant_panda' with 'giant panda'
                        elif (ann_check_list[ii].split(' ')>=min_cap_length):
                            cap_str_record = ann_check_list[ii].replace('_', ' ')
                            break
                    elif(len(cap_str.split(' '))>min_cap_length):
                        cap_str_record = cap_str.replace('_', ' ') # use annotation file without checking
                if cap_str_record is None:
                    print('invalid annotation file %s\n' %(ann_file_full_path))
                    continue
                if cap_str_record.split(' ')<min_cap_length:
                    continue
                if ins_id not in ann_dict_set_dict.keys():
                    ann_dict_set_dict[ins_id] = list()
                ann_dict_set_dict[ins_id].append(cap_str_record)
    list_for_write = list()
    key_list = ann_dict_set_dict.keys()
    key_list.sort()
    for i, ins_id in enumerate(key_list):
        list_for_write.append('%d, %s' %(ins_id, ann_dict_set_dict[ins_id][-1]))
    out_fn = os.path.join(cap_folder, 'instance_annoatation_list_v2_1.txt')
    textdump(out_fn, list_for_write)
    print('finish preparing the annotation list')


def caption_to_word_list(des_str):
    import string
    des_str = des_str.lower().replace('_', ' ').replace(',' , ' ').replace('-', ' ')
    for c in string.punctuation:
        des_str = des_str.replace(c, '')
    return split_carefully(des_str.lower().replace('_', ' ').replace('.', '').replace(',', '').replace("\'", '').replace('-', '').replace('\n', '').replace('\r', '').replace('\"', '').rstrip().replace("\\",'').replace('?', '').replace('/','').replace('#','').replace('(', '').replace(')','').replace(';','').replace('!', '').replace('/',''), ' ')

def build_vid_word_list():
    set_name_list = ['train', 'val', 'test']
    ann_cap_path = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples'
    word_list  = list()
    for i, set_name in enumerate(set_name_list):
        #ann_cap_set_fn = os.path.join(ann_cap_path, set_name+'_ann_list.txt')
        ann_cap_set_fn = os.path.join(ann_cap_path, set_name+'_ann_list_v2.txt')
        cap_lines = textread(ann_cap_set_fn)
        for ii, line in enumerate(cap_lines):
            ins_id_str, caption = line.split(',', 1)
            word_list_tmp = caption_to_word_list(caption)
            word_list += word_list_tmp
    word_list= list(set(word_list))
    return word_list 



def statistic_vid_word_list():
    set_name_list = ['train', 'val', 'test']
    ann_cap_path = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples'
    word_list  = list()
    cap_num = 0
    for i, set_name in enumerate(set_name_list):
        #ann_cap_set_fn = os.path.join(ann_cap_path, set_name+'_ann_list.txt')
        ann_cap_set_fn = os.path.join(ann_cap_path, set_name+'_ann_list_v2.txt')
        cap_lines = textread(ann_cap_set_fn)
        for ii, line in enumerate(cap_lines):
            ins_id_str, caption = line.split(',', 1)
            word_list_tmp = caption_to_word_list(caption)
            while '' in word_list_tmp:
                word_list_tmp.remove('')
                #pdb.set_trace()
            word_list += word_list_tmp
            cap_num +=1
    print('Average word length: %f\n'%(len(word_list)*1.0/cap_num)) 
    print('total word number: %f\n'%(len(word_list))) 
    word_dict = list(set(word_list))
    print('word in dictionary: %f\n'%(len(word_dict)*1.0))
    
    # get frequence
    word_to_dict ={}
    for i, word in enumerate(word_list):
        if word in word_to_dict.keys():
            word_to_dict[word] +=1
        else:
            word_to_dict[word] =1
    sorted_word = sorted(word_to_dict.items(), key=operator.itemgetter(1))
    
    sorted_word.reverse()
    
    
    
    topK = 30
    plot_data =[]
    cat_name = []
    data_fn = 'word_noun.pdf'
    count_num = 0
    for i in range(len(sorted_word)):
        if sorted_word[i][0] not in stopwords.words("english"):
            print(sorted_word[i])
            plot_data.append(sorted_word[i][1])
            cat_name.append(sorted_word[i][0])
            count_num +=1
        if count_num>=topK:
            break
    #pdb.set_trace()
    plot_data.reverse()
    cat_name.reverse()
    plot_distribution_word_ori(plot_data, cat_name, data_fn,rot=30, fsize=110)
    #pdb.set_trace()
    return word_list

def plot_distribution_word_ori(data_plot, cat_list, bar_name, rot=90, fsize=8):
    plt.close()
    #fig, ax = plt.subplots(figsize=(12, 12))
    fig = plt.figure(figsize=(180, 50))
    #fig = plt.figure()
    #frame = plt.gca()
    width =1.0
    ind = np.linspace(0, 1.3*(len(cat_list)-1), len(cat_list))
    plt.xticks(np.arange(0, ind[-1]+1))
    plt.bar(ind, data_plot, width)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.yscale('log') 
    ax.set_xlim([0-width, ind[-1]+width])
    plt.ylabel("Count",fontsize=fsize, weight='bold')
    plt.yticks(fontsize=fsize, weight='bold')
    #plt.yticks(np.arange(100,1400, 100), fontsize=fsize)
    plt.xticks(ind, cat_list, rotation=rot, fontsize=fsize, weight='bold', ha='right' )
    for a,b in zip(list(ind),data_plot):  
         plt.text(a, b+0.05, '%d' % b, ha='center', va= 'bottom',fontsize=fsize, weight='bold') 
    plt.savefig(bar_name)
    plt.show()



def plot_distribution_word(data_plot, cat_list, bar_name, rot=90, fsize=8):
    plt.close()
    #fig, ax = plt.subplots(figsize=(12, 12))
    fig = plt.figure(figsize=(120, 60))
    #fig = plt.figure()
    #frame = plt.gca()
    plt.bar(range(len(cat_list)), data_plot)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.yscale('log') 
    #plt.ylabel("Count",fontsize=fsize)
    plt.yticks(fontsize=fsize, weight='bold')
    #plt.yticks(np.arange(100,1400, 100), fontsize=fsize)
    plt.xticks(range(len(cat_list)), cat_list, rotation=rot, fontsize=fsize, weight='bold' )
    for a,b in zip(range(len(cat_list)),data_plot):  
         plt.text(a, b+0.05, '%d' % b, ha='center', va= 'bottom',fontsize=fsize, weight='bold') 
    plt.savefig(bar_name)
    plt.show()


def show_cat_distribution():
    set_name_list = ['train', 'val', 'test']
    #set_name_list = ['test']
    opt = parse_args()
    opt.dbSet = 'vid'
    cat_name_dict ={}
    #cache_fn = 'cat_cache.pk'
    cache_fn = 'cat_cache_new.pk'
    if os.path.isfile(cache_fn):
        cat_name_dict = pickleload(cache_fn)
    else:
        for set_id, set_name in enumerate(set_name_list):
            opt.set_name = set_name
            data_loader, dataset = build_dataloader(opt)
            set_length = len(dataset)
            for ins_id in range(set_length):
                index = dataset.use_key_index[ins_id]
                ann, vd_name = dataset.vid_parser.get_shot_anno_from_index(index)
                class_id =  str(ann['track'][0]['class'])
                #if class_id=='domestic_cat':
                #if class_id=='airplane':
                #if class_id=='bicycle':
                #if class_id=='zebra':
                #if class_id=='motorcycle':
                #if class_id=='elephant':
                #if class_id=='bear':
                #if class_id=='bird':
                if class_id=='watercraft':
                    print(class_id)
                    print(index)
                    pdb.set_trace()
                if class_id  in cat_name_dict.keys():
                    cat_name_dict[class_id] +=1
                else:
                    cat_name_dict[class_id] =1
       
        pickledump(cache_fn, cat_name_dict)
    
    sorted_cat = sorted(cat_name_dict.items(), key=operator.itemgetter(1))
    #sorted_cat.reverse()
    cat_name_list = []
    count_list = []
    for idx in range(len(sorted_cat)):
        class_id=sorted_cat[idx][0]
        cat_name_list.append(class_id)
        count_list.append(cat_name_dict[class_id])
    for i, cat_name in enumerate(cat_name_list):
        if cat_name=='domestic_cat':
            cat_name_list[i] = 'cat'
        if cat_name=='giant_panda':
            #cat_name_list[i] = 'giant panda'
            cat_name_list[i] = 'panda'
        if cat_name=='red_panda':
            cat_name_list[i] = 'red panda'
    data_fn = 'cat_noun_25_log.pdf'
    #data_fn = 'cat_noun_25_symlog.pdf'
    plot_distribution_cat(count_list, cat_name_list, data_fn, 25, fsize=120)
    #plot_distribution_cat_log_log(count_list, cat_name_list, data_fn, 30, fsize=120)
    #data_fn = 'cat_noun_25_linear.pdf'
    #plot_distribution_cat_linear(count_list, cat_name_list, data_fn, 30, fsize=120)
    print(cat_name_dict)


def plot_distribution_cat(data_plot, cat_list, bar_name, rot=90, fsize=8):
    plt.close()
    #fig, ax = plt.subplots(figsize=(12, 12))
    fig = plt.figure(figsize=(180, 50))
    #fig = plt.figure()
    #frame = plt.gca()
    width =1.0
    ind = np.linspace(0, 1.3*(len(cat_list)-1), len(cat_list))
    plt.xticks(np.arange(0, ind[-1]+1))
    plt.bar(ind, data_plot, width)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.yscale('log') 
    ax.set_xlim([0-width, ind[-1]+width])
    plt.ylabel("Count",fontsize=fsize, weight='bold')
    plt.yticks(fontsize=fsize, weight='bold')
    #plt.yticks(np.arange(100,1400, 100), fontsize=fsize)
    plt.xticks(ind, cat_list, rotation=rot, fontsize=fsize, weight='bold', ha='right' )
    for a,b in zip(list(ind),data_plot):  
         plt.text(a, b+0.05, '%d' % b, ha='center', va= 'bottom',fontsize=fsize, weight='bold') 
    plt.savefig(bar_name)
    plt.show()


def plot_distribution_cat_log_log(data_plot, cat_list, bar_name, rot=90, fsize=8):
    plt.close()
    #fig, ax = plt.subplots(figsize=(12, 12))
    fig = plt.figure(figsize=(180, 55))
    #fig = plt.figure()
    #frame = plt.gca()
    plt.bar(range(len(cat_list)), data_plot)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.yscale('symlog') 
    #plt.ylabel("Count",fontsize=fsize)
    plt.yticks(fontsize=fsize, weight='bold')
    #plt.yticks(np.arange(100,1400, 100), fontsize=fsize)
    plt.xticks(range(len(cat_list)), cat_list, rotation=rot, fontsize=fsize, weight='bold' )
    for a,b in zip(range(len(cat_list)),data_plot):  
         plt.text(a, b+0.05, '%d' % b, ha='center', va= 'bottom',fontsize=fsize, weight='bold') 
    plt.savefig(bar_name)
    plt.show()


def plot_distribution_cat_linear(data_plot, cat_list, bar_name, rot=90, fsize=8):
    plt.close()
    #fig, ax = plt.subplots(figsize=(12, 12))
    fig = plt.figure(figsize=(180, 55))
    #fig = plt.figure()
    #frame = plt.gca()
    plt.bar(range(len(cat_list)), data_plot)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    #plt.yscale('symlog') 
    #plt.ylabel("Count",fontsize=fsize)
    plt.yticks(fontsize=fsize, weight='bold')
    #plt.yticks(np.arange(100,1400, 100), fontsize=fsize)
    plt.xticks(range(len(cat_list)), cat_list, rotation=rot, fontsize=fsize, weight='bold' )
    for a,b in zip(range(len(cat_list)),data_plot):  
         plt.text(a, b+0.05, '%d' % b, ha='center', va= 'bottom',fontsize=fsize, weight='bold') 
    plt.savefig(bar_name)
    plt.show()

def multi_process_get_item_cache(param):
    dataset_loader, index = param
    print(index)
    out_list = dataset_loader.__getitem__(index)

def get_set_visual_feature_cache():
    opt = parse_args()
    opt.set_name = 'train'
    opt.dbSet = 'vid'
    cpu_num = 20
    #pdb.set_trace()
    data_loader, dataset = build_dataloader(opt)
    pdb.set_trace()
    set_length = len(dataset.vid_parser.tube_cap_dict)
    index_list = list(range(set_length))
    param_list = list()
    for index in index_list:
        param_list.append([dataset, index])
    pdb.set_trace()
    for stIdx in range(0, len(param_list), cpu_num):
        edIdx = stIdx + cpu_num
        if edIdx>len(param_list):
            edIdx = len(param_list)
        parmap(multi_process_get_item_cache, param_list[stIdx: edIdx])
    
def get_h5_feature_dict(h5file_path):
    img_prp_reader = h5py.File(h5file_path, 'r')
    return img_prp_reader

def check_str_valid():
    train_ann_list_fn_only = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/train_ann_list_v2_txt_only.txt'  
    val_ann_list_fn_only = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_ann_list_v2_txt_only.txt'
    txt_lines = textread(val_ann_list_fn_only)
    for i, txt_line in enumerate(txt_lines):
        ele_list = txt_line.split(' ')
        #pdb.set_trace()
        print(i)
        if len(ele_list)<2:
            pdb.set_trace()

def sample_frames_from_vid_parser(vid_parper, index ,save_num, out_fd, color_map=(0,0,255)):
    ann, vdName =vid_parper.get_shot_anno_from_index(index)
    track = ann['track']
    trackId = ann['id']
    #print(track)
    im_height = 300
    imFd = os.path.join(vid_parper.jpg_folder, vdName) 
    tmpList = list()
    frmNum = 36
    if frmNum>len(track):
        frmNum = len(track)
    smp_width = int(math.floor(len(track)*1.0/frmNum))
    save_width = int(frmNum/save_num)
    for iii in range(frmNum):
        frmId = smp_width*iii
        vdFrmInfo = track[frmId]
        imPath = imFd + '/' + '%06d.JPEG' %(vdFrmInfo['frame']-1)
        img = cv2.imread(imPath)
        bbox = tuple(vdFrmInfo['bbox'])
        imOut = cv2.rectangle(img, bbox[0:2], bbox[2:4], color_map, 6)
        hs, ws, cs = imOut.shape
        imOut = cv2.resize(imOut, (int(ws*1.0*im_height/hs), im_height))
        imNameRaw = os.path.basename(imPath).split('.')[0]
        vdNameOut = vdName.replace('/', '__')
        imOutPath = os.path.join(out_fd, 'samples', str(index) ,
                imNameRaw + '_' +vdNameOut + '_' + str(frmId)  + '.JPEG')
        makedirs_if_missing(os.path.dirname(imOutPath))
        tmpList.append(imOut)
        if(iii%save_width==0):
            cv2.imwrite(imOutPath, imOut)
    print('finish generating set: %d\n' %(index))

def sample_for_paper():
    #ins_list = [250, 3889, 311, 1646]
    #ins_list = [372]
    #ins_list = [677]
    #ins_list = [2012]
    #ins_list =[119]
    #ins_list =[64]
    #ins_list =[393]
    #ins_list =[195]
    #ins_list =[286]
    #ins_list =[175]
    #ins_list =[2066]
    #ins_list =[1902]
    #ins_list =[22]
    #ins_list =[950]
    ins_list =[2075]
    #color_map = (0, 0, 255) # red
    color_map = (0, 255, 0) # red
    opt = parse_args()
    opt.set_name = 'train'
    opt.dbSet = 'vid'
    save_num = 20
    out_fd ='../data/figure_paper'
    data_loader, dataset = build_dataloader(opt)
    vid_parper = dataset.vid_parser
    for i, index in enumerate(ins_list):
        sample_frames_from_vid_parser(vid_parper, index ,save_num, out_fd, color_map)

def draw_specific_tube_proposals(vid_parser, index, tube_id_list, tube_proposal_list, out_fd, color_list=None):
    
    load_image_flag = True
    lbl = index
    frmImList = list()
    tube_info_sub_prp, frm_info_list = tube_proposal_list
    if color_list is None:
        color_list =[(255, 0, 0), (0, 255, 0)]
        #color_list =[(0, 255, 0), (255, 0, 0)]
    dotted = False
    line_width = 6
    for ii, tube_id in enumerate(tube_id_list):
        if ii==1:
            dotted = True
            line_width =3
        tube = copy.deepcopy(tube_info_sub_prp[0][tube_id])

        if load_image_flag:
            # visualize sample results
            vd_name, ins_id_str = vid_parser.get_shot_info_from_index(lbl)
            frmImNameList = [os.path.join(vid_parser.jpg_folder, vd_name, frame_name + '.JPEG') for frame_name in frm_info_list]
            for fId, imPath  in enumerate(frmImNameList):
                img = cv2.imread(imPath)
                frmImList.append(img)
            vis_frame_num = 3000
            visIner =max(int(len(frmImList) /vis_frame_num), 1)
            load_image_flag = False
            
            frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
        
        tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
        print('visualizing tube %d\n'%(tube_id))
        tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
        frmImList_vis = vis_image_bbx(frmImList_vis, tube_vis_resize, color_list[ii], line_width, dotted)
        #frmImList_vis = vis_gray_but_bbx(frmImList_vis, tube_vis_resize)
        break

    out_fd_full = os.path.join(out_fd, vid_parser.set_name + str(lbl))
    makedirs_if_missing(out_fd_full)
    frm_name_list = list()
    for i, idx  in enumerate(range(0, len(frmImList), visIner)):
        out_fn_full = os.path.join(out_fd_full, frm_info_list[idx]+'.jpg')
        cv2.imwrite(out_fn_full, frmImList_vis[i])
        frm_name_list.append(frm_info_list[idx])
    return frmImList_vis, frm_name_list  

def vis_compare_ins():
    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name = 'val'
    out_fd ='../data/figure_paper/compare_sample'
    ins_id =[528, 626]
    data_loader, dataset = build_dataloader(opt)
    vid_parser = dataset.vid_parser
    tube_id_list=[2, 20]
    for i, index in enumerate(ins_id):
        ins_ann, vd_name = vid_parser.get_shot_anno_from_index(index)
        tube_info_path = os.path.join(dataset.tubePath, dataset.set_name, dataset.prp_type, str(index)+'.pd') 
        tube_proposal_list = pickleload(tube_info_path)
        draw_specific_tube_proposals(vid_parser, index, tube_id_list, tube_proposal_list, out_fd)

def parse_tube_info(tube_info_fn):
    info_list = textread(tube_info_fn)
    tube_info_obj = {}
    ins_list = list()
    tube_id_att = list()
    tube_id_dvsa = list()
    tube_id_gr = list()
    tube_ov_att = list()
    tube_ov_dvsa = list()
    tube_ov_gr = list()
    tube_ov_frm = list()
    for i, line_info in enumerate(info_list):
        text_segments = line_info.split(' ')
        ins_list.append(int(text_segments[0]))
        tube_id_att.append(int(text_segments[2]))
        tube_ov_att.append(float(text_segments[3]))
        tube_id_dvsa.append(int(text_segments[4]))
        tube_ov_dvsa.append(float(text_segments[5]))
        tube_id_gr.append(int(text_segments[6]))
        tube_ov_gr.append(float(text_segments[7]))
        tube_ov_frm.append(float(text_segments[8]))

    tube_info_obj['ins_list'] = ins_list
    tube_info_obj['tube_id_att'] = tube_id_att
    tube_info_obj['tube_ov_att'] = tube_ov_att
    tube_info_obj['tube_id_dvsa'] = tube_id_dvsa
    tube_info_obj['tube_ov_dvsa'] = tube_ov_dvsa
    tube_info_obj['tube_id_gr'] = tube_id_gr
    tube_info_obj['tube_ov_gr'] = tube_ov_gr
    tube_info_obj['tube_ov_frm'] = tube_ov_frm
    return tube_info_obj


def vis_compare_ins_val():
    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name = 'val'
    out_fd ='../data/figure_paper/compare_sample'
    tube_info_fn = '../data/figure_paper/compare_sample/val_tube_list.txt'
    data_loader, dataset = build_dataloader(opt)
    vid_parser = dataset.vid_parser
    # tube_id_list=[2, 20]
    tube_info_obj = parse_tube_info(tube_info_fn)
    ins_id_list = tube_info_obj['ins_list']
    tube_id_att = tube_info_obj['tube_id_att']
    tube_id_dvsa = tube_info_obj['tube_id_dvsa']
    tube_id_gr = tube_info_obj['tube_id_gr']

    for i, index in enumerate(ins_id_list):
        ins_ann, vd_name = vid_parser.get_shot_anno_from_index(index)
        tube_id_list = [tube_id_att[i], tube_id_dvsa[i], tube_id_gr[i]]
        tube_info_path = os.path.join(dataset.tubePath, dataset.set_name, dataset.prp_type, str(index)+'.pd') 
        tube_proposal_list = pickleload(tube_info_path)
        pdb.set_trace()
        draw_specific_tube_proposals(vid_parser, index, tube_id_list, tube_proposal_list, out_fd)

def get_gt_bbx(ins_ann):
    tube_length = len(ins_ann['track'])
    gt_bbx = list()
    for i in range(tube_length):
        tmp_bbx = copy.deepcopy(ins_ann['track'][i]['bbox'])
        h, w = ins_ann['track'][i]['frame_size']
        tmp_bbx[0] = tmp_bbx[0]*1.0/w
        tmp_bbx[2] = tmp_bbx[2]*1.0/w
        tmp_bbx[1] = tmp_bbx[1]*1.0/h
        tmp_bbx[3] = tmp_bbx[3]*1.0/h
        gt_bbx.append(tmp_bbx)
    return gt_bbx

def get_frm_result(result_frm, index):
    for result_tmp in result_frm:
        index_tmp = result_tmp[0]
        if index_tmp==index:
            return result_tmp[2][0][0][0]

def vis_compare_ins_val_sep():
    opt = parse_args()
    opt.dbSet = 'vid'
    out_fd ='../data/figure_paper/compare_sample'
    opt.set_name = 'val'
    tube_info_fn = '../data/figure_paper/compare_sample/val_tube_list_ver_dense.txt'
    #tube_info_fn = '../data/figure_paper/compare_sample/val_tube_list_v2.txt'
    frm_result_fn = '../data/final_models/tube_result/_bs_16_tn_30_wl_20_cn_1_fd_512_rankFrm_fc_none_full_txt_gru_rgb_lr_0.0_vid_margin_10.0_frm_level_result_val_rankFrm_ep_5_itr0.pk'
    #tube_info_fn = '../data/figure_paper/compare_sample/test_tube_list_ver_choose.txt'
    #opt.set_name = 'test'
    result_frm = pickleload(frm_result_fn)
    #pdb.set_trace()
    
    data_loader, dataset = build_dataloader(opt)
    vid_parser = dataset.vid_parser
    # tube_id_list=[2, 20]
    tube_info_obj = parse_tube_info(tube_info_fn)
    ins_id_list = tube_info_obj['ins_list']
    tube_id_att = tube_info_obj['tube_id_att']
    tube_id_dvsa = tube_info_obj['tube_id_dvsa']
    tube_id_gr = tube_info_obj['tube_id_gr']

    tube_ov_att = tube_info_obj['tube_ov_att']
    tube_ov_dvsa = tube_info_obj['tube_ov_dvsa']
    tube_ov_gr = tube_info_obj['tube_ov_gr']
    tube_ov_frm = tube_info_obj['tube_ov_frm']
    #pdb.set_trace()
    for i, index in enumerate(ins_id_list):
        ins_ann, vd_name = vid_parser.get_shot_anno_from_index(index)
        tube_info_path = os.path.join(dataset.tubePath, dataset.set_name, dataset.prp_type, str(index)+'.pd') 
        tube_proposal_list = pickleload(tube_info_path)
        
        # adding gt to prp
        tube_gt = get_gt_bbx(ins_ann)
        tube_prp_id_length = len(tube_proposal_list[0][0])
        tube_proposal_list[0][0].append(tube_gt)
        tube_proposal_list[0][1].append(1)
        # adding frm prp

        tube_proposal_list[0][0].append(get_frm_result(result_frm, index))
        tube_proposal_list[0][1].append(tube_ov_frm[i])
       
        tube_id_list_ori = [tube_id_att[i], tube_id_dvsa[i], tube_id_gr[i], tube_prp_id_length+1]

        im_list = list()
        for j in range(4):
            new_tube_id_list = [tube_id_list_ori[j], tube_prp_id_length]
            #new_tube_id_list = [tube_prp_id_length, tube_id_list_ori[j] ]
            out_fd_sub = os.path.join(out_fd, str(j))
            frmImList_vis, frm_name_list = draw_specific_tube_proposals(vid_parser, index, new_tube_id_list, copy.deepcopy(tube_proposal_list), out_fd_sub)
            im_list.append(frmImList_vis)

        for im_id, frm_name in enumerate(frm_name_list):
            mat_att = im_list[0][im_id]
            mat_dvsa = im_list[1][im_id]
            mat_gr = im_list[2][im_id]
            mat_frm = im_list[3][im_id]
            h, w, c = mat_att.shape
            mat_all = np.zeros((h*2, w*2, c), np.float32)
            mat_all[:h, :w, :] = mat_att
            mat_all[:h, w:2*w, :] = mat_dvsa
            mat_all[h:2*h, :w, :] = mat_gr
            mat_all[h:2*h, w:2*w, :] = mat_frm
            out_fd_sub = os.path.join(out_fd, 'full', dataset.set_name+'_'+str(index), frm_name +'.jpg')
            makedirs_if_missing(os.path.dirname(out_fd_sub))
            cv2.imwrite(out_fd_sub, mat_all)

        print('finish')


def sample_best_tube():
    att_result_fn ='../data/final_models/tube_result/log_bs_4_tn_30_wl_20_cn_1_fd_512_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_margin_10.0lstm_hd_512result_val_coAttV1_ep_21_lamda_1.pk'
    dvsa_result_fn ='../data/final_models/tube_result/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_lstmV2_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.0result_val_rank_lstm_ep_30_lamda_0.pk'
    gr_result_fn ='../data/final_models/tube_result/_bs_16_tn_30_wl_20_cn_1_fd_512_rankGroundRV2_lstmV2_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.0result_val_gr_lstm_ep_26_lamda_0.pk'
    frm_result_fn = '../data/final_models/tube_result/_bs_16_tn_30_wl_20_cn_1_fd_512_rankFrm_fc_none_full_txt_gru_rgb_lr_0.0_vid_margin_10.0_frm_level_result_val_rankFrm_ep_5_itr0.pk'
    
    #att_result_fn ='../data/final_models/tube_result/tube_result_bs_4_tn_30_wl_20_cn_1_fd_512_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_margin_10.0lstm_hd_512result_test_coAttV1_ep_21_lamda_1.pk'
    #dvsa_result_fn ='../data/final_models/tube_result/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_lstmV2_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.0result_test_rank_lstm_ep_30_lamda_0.pk'
    #gr_result_fn ='../data/final_models/tube_result/_bs_16_tn_30_wl_20_cn_1_fd_512_rankGroundRV2_lstmV2_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.0result_test_gr_lstm_ep_26_lamda_0.pk'
    #frm_result_fn = '../data/final_models/tube_result/_bs_16_tn_30_wl_20_cn_1_fd_512_rankFrm_fc_none_full_txt_gru_rgb_lr_0.0_vid_margin_10.0_frm_level_result_test_rankFrm_ep_5_itr0.pk'
     
    result_att = pickleload(att_result_fn)
    result_dvsa = pickleload(dvsa_result_fn)
    result_gr = pickleload(gr_result_fn)
    result_frm = pickleload(frm_result_fn)
    
    thre_id =4 # 0.5
    thre_list_length=len(result_att)
    test_set_length = len(result_att[thre_id])

    #pdb.set_trace()
    txt_out_fn = '../data/figure_paper/compare_sample/val_tube_list_ver_choose_v4.txt'
    #txt_out_fn = '../data/figure_paper/compare_sample/test_tube_list_ver_choose.txt'
    #txt_out_fn = '../data/figure_paper/compare_sample/test_tube_list.txt'
    sample_list = list()
    for idx in range(test_set_length):
        cur_att_smp = result_att[thre_id][idx]
        cur_dvsa_smp = result_dvsa[thre_id][idx]
        cur_gr_smp = result_gr[thre_id][idx]
        if cur_att_smp[0]==23:
            pdb.set_trace()
        #if (cur_att_smp[1]>0 and cur_dvsa_smp[1]<1 and cur_gr_smp[1]<1 and result_frm[idx][1]<1):
        if (0 and cur_att_smp[1]>0 and cur_dvsa_smp[1]<1 and cur_gr_smp[1]<1):
        #if (cur_att_smp[1]>0  and result_frm[idx][1]<1):
        #if (cur_att_smp[1]>0 ):
            print(cur_att_smp[:2])
            for i in range(thre_list_length):
                print(result_dvsa[i][idx][:2])
                print(result_gr[i][idx][:2])
            print(cur_att_smp[0]+7742)
            tmp_shot_info = '%d %d %d %f %d %f %d %f %f' %(cur_att_smp[0], cur_att_smp[0]+7742 , \
                    cur_att_smp[2][0], cur_att_smp[4][0], cur_dvsa_smp[2][0], cur_dvsa_smp[4][0], \
                    cur_gr_smp[2][0], cur_gr_smp[4][0], result_frm[idx][3][0]) 
            sample_list.append(tmp_shot_info)
            #pdb.set_trace()
    textdump(txt_out_fn, sample_list) 

def distribute_compare():
    att_result_fn ='../data/final_models/tube_result/log_bs_4_tn_30_wl_20_cn_1_fd_512_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_margin_10.0lstm_hd_512result_val_coAttV1_ep_21_lamda_1.pk'
    att_result_abs_fn ='../data/final_models/tube_result/log_bs_4_tn_30_wl_20_cn_1_fd_512_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_margin_10.0lstm_hd_512result_val_coAttV1_ep_21_lamda_1.pk'
    result_att = pickleload(att_result_fn)
    result_att_abs = pickleload(att_result_abs_fn)




def visualize_caption_weight():
    cap_w = 0.001*[-1.7339, -1.1330,  0.9425, -0.4879, -1.0536, -1.6465, -1.4925,]
    cap_name = ['black', 'white', 'puppy', 'in', 'the', 'middle', 'is', 'eating', 'food', 'in', 'mans', 'hand']



if __name__ == '__main__':
    #vis_compare_ins()
    #vis_compare_ins_val_sep()
    #sample_best_tube()
    
    #sample_for_paper()
    #statistic_vid_word_list()
    #show_cat_distribution()
    #check_str_valid()
    #vid_split_validation_test()
    #vid_txt_only()

    #file_path = '/data1/zfchen/code/video_feature/feature_extraction/tmp/vid/val/0.h5'
    #out_dict = get_h5_feature_dict(file_path)
    #pdb.set_trace()
    #get_set_visual_feature_cache() 
    #vid_caption_processing()
    #vid_valid_caption_preprocessing()
    #statistic_im_prp()
    #vis_im_prp()
    #test_im_prp()
    #pdb.set_trace()


