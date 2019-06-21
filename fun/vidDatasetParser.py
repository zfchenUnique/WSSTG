
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
from fun.datasetLoader import *
import math

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

def get_h5_feature_dict(h5file_path):
    img_prp_reader = h5py.File(h5file_path, 'r')
    return img_prp_reader

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


if __name__ == '__main__':
    pdb.set_trace()


