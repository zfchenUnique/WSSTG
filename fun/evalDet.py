#import os
import numpy as np
import sys
import magic
import re
sys.path.append('..')
from util.mytoolbox import *
from util.get_image_size import get_image_size
import cv2
import pdb
import ipdb
import copy
import torch
from fun.datasetLoader import *
from vidDatasetParser import evaluate_tube_recall_vid, resize_tube_bbx 
from netUtil import *

sys.path.append('../annotation')
from script_test_annotation import evaluate_tube_recall

def computeIoU(box1, box2):
  # each box is of [x1, y1, w, h]
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
  inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = box1[2]*box1[3] + box2[2]*box2[3] - inter
  return float(inter)/union

# get original size of the proposls
# the proposals are extracted at larger size
def transFormBbx(bbx, im_shape,reSize=800, maxSize=1200, im_scale=None):
    if im_scale is not None:
        bbx = [ ele/im_scale for ele in bbx]
        bbx[2] = bbx[2]-bbx[0]
        bbx[3] = bbx[3]-bbx[1]
        return bbx
    im_size_min = np.min(im_shape[0:2]) 
    im_size_max = np.max(im_shape[0:2]) 
    im_scale = float(reSize)/im_size_min
    if(np.round(im_size_max*im_scale) > maxSize):
        im_scale = float(maxSize)/ float(im_size_max)
    bbx = [ ele/im_scale for ele in bbx]
    bbx[2] = bbx[2]-bbx[0]
    bbx[3] = bbx[3]-bbx[1]
    return bbx
    

class evalDetAcc(object):
    def __init__(self, gtBbxList=None, IoU=0.5, topK=1):
        self.gtList = gtBbxList
        self.thre = IoU
        self.topK= topK
    
    def evalList(self, prpList):
        imgNum = len(prpList)
        posNum = 0
        for i in range(imgNum):
            tmpRpList= prpList[i]
            gtBbx = self.gtList[i]
            for j in range(self.topK):
                iou = computeIoU(gtBbx, tmpRpList[j])
                if iou>self.thre:
                    posNum +=1
                    break
        return float(posNum)/imgNum

def rpMatPreprocess(rpMatrix, imWH, isA2D=False):
    rpList= list()
    rpNum = rpMatrix.shape[0]
    for i in range(rpNum):
        tmpRp = list(rpMatrix[i, :])
        if not isA2D:
            bbx = transFormBbx(tmpRp, imWH)
        else:
            bbx = transFormBbx(tmpRp, imWH, im_scale=imWH) 
        rpList.append(bbx)
    return rpList

def evalAcc_att_test(simMM_batch, tubeInfo, indexOri, datasetOri, visRsFd, visFlag=False, topK = 1, thre_list=[0.5], more_detailed_flag=False):
    resultList= list()
    bSize = len(indexOri)
    tube_Prp_num = len(tubeInfo[0][0][0])
    stIdx = 0
    vid_parser = datasetOri.vid_parser
    for idx, lbl in enumerate(indexOri):
        simMM = simMM_batch[idx, :]
        simMMReshape = simMM.view(-1, tube_Prp_num) 
        
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
        sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)

        tube_Info_sub = tubeInfo[idx]
        tube_info_sub_prp, frm_info_list = tube_Info_sub
        tube_info_sub_prp_bbx, tube_info_sub_prp_score = tube_info_sub_prp
        #prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]], sort_sim_np[i] ]for i in range(topK)]
        prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]] for i in range(topK)], [sort_sim_np[i] for i in range(topK)] ]
        shot_proposals = [prpListSort, frm_info_list]
        for ii, thre in enumerate(thre_list):
            if more_detailed_flag:
                recall_tmp, iou_list = evaluate_tube_recall_vid(shot_proposals, vid_parser, lbl, thre, topKOri=topK, more_detailed_flag=more_detailed_flag)
            else:
                recall_tmp= evaluate_tube_recall_vid(shot_proposals, vid_parser, lbl, thre, topKOri=topK)
        if more_detailed_flag:
            resultList.append((lbl, recall_tmp[-1], simIdx, sort_sim_np, iou_list))
        else:
            resultList.append((lbl, recall_tmp[-1]))
        print('accuracy for %d: %3f' %(lbl, recall_tmp[-1]))
        
        if visFlag:
            
            print(vid_parser.tube_cap_dict[lbl])
            print(lbl)
            #continue
            vd_name, ins_id_str = vid_parser.get_shot_info_from_index(lbl)
            frmImNameList = [os.path.join(vid_parser.jpg_folder, vd_name, frame_name + '.JPEG') for frame_name in frm_info_list]
            frmImList = list()
            for fId, imPath  in enumerate(frmImNameList):
                img = cv2.imread(imPath)
                frmImList.append(img)
            vis_frame_num = 30
            visIner =max(int(len(frmImList) /vis_frame_num), 1)
            
            for ii in range(topK):
                print('visualizing tube %d\n'%(ii))
                #pdb.set_trace() 
                tube = prpListSort[0][ii]
                frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
                tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
                tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
                vd_name_raw = vd_name.split('/')[-1]
                makedirs_if_missing(visRsFd)
                visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis_resize, visRsFd+'/'+vd_name_raw+ '_' + str(ii)+'.gif')
            pdb.set_trace()
    return resultList


def evalAcc_att(simMMFull, tubeInfo, indexOri, datasetOri, visRsFd, visFlag=False, topK = 1, thre_list=[0.5], more_detailed_flag=False):
#    pdb.set_trace()
    resultList= list()
    bSize = len(indexOri)
    tube_Prp_num = len(tubeInfo[0][0][0])
    stIdx = 0
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    vid_parser = datasetOri.vid_parser
    for idx, lbl in enumerate(indexOri):
        simMM = simMMFull[idx, :, idx]
        simMMReshape = simMM.view(-1, tube_Prp_num) 
        #pdb.set_trace()
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
        sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)

        tube_Info_sub = tubeInfo[idx]
        tube_info_sub_prp, frm_info_list = tube_Info_sub
        tube_info_sub_prp_bbx, tube_info_sub_prp_score = tube_info_sub_prp
        #prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]], sort_sim_np[i] ]for i in range(topK)]
        prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]] for i in range(topK)], [sort_sim_np[i] for i in range(topK)] ]
        shot_proposals = [prpListSort, frm_info_list]
        for ii, thre in enumerate(thre_list):
            if more_detailed_flag:
                recall_tmp, iou_list = evaluate_tube_recall_vid(shot_proposals, vid_parser, lbl, thre, topKOri=topK, more_detailed_flag=more_detailed_flag)
            else:
                recall_tmp= evaluate_tube_recall_vid(shot_proposals, vid_parser, lbl, thre, topKOri=topK)
        if more_detailed_flag:
            resultList.append((lbl, recall_tmp[-1], simIdx, sort_sim_np, iou_list))
        else:
            resultList.append((lbl, recall_tmp[-1]))
        print('accuracy for %d: %3f' %(lbl, recall_tmp[-1]))
        
        #pdb.set_trace()
        if visFlag:
            

            # visualize sample results
            #if(recall_tmp[-1]<=0.5):
            #    continue
            print(vid_parser.tube_cap_dict[lbl])
            print(lbl)
            #pdb.set_trace()
            #continue
            vd_name, ins_id_str = vid_parser.get_shot_info_from_index(lbl)
            frmImNameList = [os.path.join(vid_parser.jpg_folder, vd_name, frame_name + '.JPEG') for frame_name in frm_info_list]
            frmImList = list()
            for fId, imPath  in enumerate(frmImNameList):
                img = cv2.imread(imPath)
                frmImList.append(img)
            vis_frame_num = 30
            visIner =max(int(len(frmImList) /vis_frame_num), 1)
            
            for ii in range(topK):
                print('visualizing tube %d\n'%(ii))
                #pdb.set_trace() 
                tube = prpListSort[0][ii]
                frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
                tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
                tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
                vd_name_raw = vd_name.split('/')[-1]
                makedirs_if_missing(visRsFd)
                visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis_resize, visRsFd+'/'+vd_name_raw+ '_' + str(ii)+'.gif')
            pdb.set_trace()
    return resultList




def evalAcc(imFtr, txtFtr, tubeInfo, indexOri, datasetOri, visRsFd, visFlag=False, topK = 1, thre_list=[0.5], more_detailed_flag=False):
    resultList= list()
    bSize = len(indexOri)
    tube_Prp_num = imFtr.shape[1]
    stIdx = 0
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    vid_parser = datasetOri.vid_parser
    assert txtFtr.shape[1]==1
    for idx, lbl in enumerate(indexOri):
        imFtrSub = imFtr[idx]
        txtFtrSub = txtFtr[idx].view(-1,1)
        simMM = torch.mm(imFtrSub, txtFtrSub)
        #pdb.set_trace()                
        simMMReshape = simMM.view(-1, tube_Prp_num) 
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
        sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)

        tube_Info_sub = tubeInfo[idx]
        tube_info_sub_prp, frm_info_list = tube_Info_sub
        tube_info_sub_prp_bbx, tube_info_sub_prp_score = tube_info_sub_prp
        #prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]], sort_sim_np[i] ]for i in range(topK)]
        prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]] for i in range(topK)], [sort_sim_np[i] for i in range(topK)] ]
        shot_proposals = [prpListSort, frm_info_list]
        for ii, thre in enumerate(thre_list):
            if more_detailed_flag:
                recall_tmp, iou_list= evaluate_tube_recall_vid(shot_proposals, vid_parser, lbl, thre, topKOri=topK, more_detailed_flag=more_detailed_flag)
            else:
                recall_tmp =  evaluate_tube_recall_vid(shot_proposals, vid_parser, lbl, thre, topKOri=topK)
        if more_detailed_flag:
            resultList.append((lbl, recall_tmp[-1], simIdx, sort_sim_np, iou_list))
        else:
            resultList.append((lbl, recall_tmp[-1]))
        #print('accuracy for %d: %3f' %(lbl, recall_tmp[-1]))
        
        #pdb.set_trace()
        if visFlag:
            # visualize sample results
            #if(recall_tmp[-1]<=0.5):
             #    continue
            vd_name, ins_id_str = vid_parser.get_shot_info_from_index(lbl)
            frmImNameList = [os.path.join(vid_parser.jpg_folder, vd_name, frame_name + '.JPEG') for frame_name in frm_info_list]
            frmImList = list()
            for fId, imPath  in enumerate(frmImNameList):
                img = cv2.imread(imPath)
                frmImList.append(img)
            vis_frame_num = 30
            visIner =max(int(len(frmImList) /vis_frame_num), 1)
            
            for ii in range(topK):
                print('visualizing tube %d\n'%(ii))
                #pdb.set_trace() 
                tube = prpListSort[0][ii]
                frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
                tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
                tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
                vd_name_raw = vd_name.split('/')[-1]
                makedirs_if_missing(visRsFd)
                visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis_resize, visRsFd+'/'+vd_name_raw+ '_' + str(ii)+'.gif')
            pdb.set_trace()
    return resultList

if __name__=='__main__':
    pdb.set_trace()
