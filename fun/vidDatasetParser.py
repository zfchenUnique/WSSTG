
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
set_debugger()

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
        self.tube_ann_list_fn = os.path.join(annFd, 'Data/VID/annSamples/', set_name + '_ann_list.txt')
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

def evaluate_tube_recall_vid(shot_proposals, vid_parser, tube_index, thre=0.5 ,topKOri=20):
    topK = min(topKOri, len(shot_proposals[0][0]))
    recall_k = [0.0] * (topK + 1)
   
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

    #pdb.set_trace()
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
            continue
        else:
            recall_k[i+1] += 1.0
            is_instance_annotated = True
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



def vid_caption_processing():
    cap_folder = '/data1/zfchen/data/ILSVRC/capResults'
    train_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/train_valid_list.txt'  
    val_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_valid_list.txt'
    ins_cap_list_fn = '/data1/zfchen/data/ILSVRC/capResults/instance_annoatation_list_v1_check.txt'
    train_ann_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/train_ann_list.txt'  
    val_ann_list_fn = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples/val_ann_list.txt'
    
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
    out_fn = os.path.join(cap_folder, 'instance_annoatation_list_v1_2.txt')
    textdump(out_fn, list_for_write)
    print('finish preparing the annotation list')


def caption_to_word_list(des_str):
    import string
    des_str = des_str.lower().replace('_', ' ').replace(',' , ' ').replace('-', ' ')
    for c in string.punctuation:
        des_str = des_str.replace(c, '')
    return split_carefully(des_str.lower().replace('_', ' ').replace('.', '').replace(',', '').replace("\'", '').replace('-', '').replace('\n', '').replace('\r', '').replace('\"', '').rstrip().replace("\\",'').replace('?', '').replace('/','').replace('#','').replace('(', '').replace(')','').replace(';','').replace('!', '').replace('/',''), ' ')

def build_vid_word_list():
    set_name_list = ['train', 'val']
    ann_cap_path = '/data1/zfchen/data/ILSVRC/Data/VID/annSamples'
    word_list  = list()
    for i, set_name in enumerate(set_name_list):
        ann_cap_set_fn = os.path.join(ann_cap_path, set_name+'_ann_list.txt')
        cap_lines = textread(ann_cap_set_fn)
        for ii, line in enumerate(cap_lines):
            ins_id_str, caption = line.split(',', 1)
            word_list_tmp = caption_to_word_list(caption)
            word_list += word_list_tmp
    word_list= list(set(word_list))
    return word_list 

def multi_process_get_item_cache(param):
    dataset_loader, index = param
    print(index)
    out_list = dataset_loader.__getitem__(index)

def get_set_visual_feature_cache():
    opt = parse_args()
    opt.set_name = 'train'
    opt.dbSet = 'vid'
    cpu_num = 20
    pdb.set_trace()
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
    
    #pool = Pool(processes=20)
    #pool.map(multi_process_get_item_cache, param_list)
    #pool.join()
    #pool.close()
    


if __name__ == '__main__':
    #get_set_visual_feature_cache() 
    #vid_caption_processing()
    #vid_valid_caption_preprocessing()
    #statistic_im_prp()
    vis_im_prp()
    test_im_prp()
    #pdb.set_trace()


