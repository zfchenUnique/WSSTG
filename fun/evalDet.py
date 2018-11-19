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

def vis_detections(im, dets, thresh=0.5, topK=20, color=0):
    """Visual debugging of detections."""

    if color==0:
        colorInfo =(0, 0, 255)
    else:
        colorInfo =(255, 255, 255)

    for i in range(np.minimum(topK, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], colorInfo, 2)
            cv2.putText(im, '%.3f' % (score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, colorInfo, thickness=1)
    return im

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

def evalAcc_frm(imFtr, txtFtr, tubeInfo, indexOri, datasetOri, frm_idx_list, bbx_list, visRsFd, visFlag=False, topK = 1, thre_list=[0.5]):
    resultList= list()
    bSize = len(indexOri)
    tube_Prp_num = imFtr.shape[1]
    stIdx = 0
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    vid_parser = datasetOri.vid_parser
    assert txtFtr.shape[1]==1
    for idx, lbl in enumerate(indexOri):
        ann, vid_name = vid_parser.get_shot_anno_from_index(lbl) 
        imFtrSub = imFtr[idx]
        txtFtrSub = txtFtr[idx].view(-1,1)
        simMM = torch.mm(imFtrSub, txtFtrSub)
        #pdb.set_trace()                
        simMMReshape = simMM.view(-1, tube_Prp_num) 
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
        sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)
        
        frm_id = frm_idx_list[idx][0] 
        tmp_prp_bbx_list = bbx_list[idx][0]
        bbx = ann['track'][frm_id]['bbox']
        h, w= ann['track'][frm_id]['frame_size']
        bbx[0] = bbx[0]*1.0/w
        bbx[2] = bbx[2]*1.0/w
        bbx[1] = bbx[1]*1.0/h
        bbx[3] = bbx[3]*1.0/h
        
        ov = compute_IoU_v2(tmp_prp_bbx_list[simIdx[0]], bbx)
        if ov> thre_list[0]:
            resultList.append([lbl, 1])
        else:
            resultList.append([lbl, 0])
    return resultList

def evalAcc_frm_tube(imFtr, txtFtr, tubeInfo, indexOri, datasetOri, frm_idx_list, bbx_list, visRsFd, visFlag=False, topK = 1, thre_list=[0.5], more_detailed_flag=False):
    resultList= list()
    bSize = len(indexOri)
    tube_Prp_num = imFtr.shape[1]
    stIdx = 0
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    vid_parser = datasetOri.vid_parser
    assert txtFtr.shape[1]==1
    assert len(indexOri)==1
    for idx, lbl in enumerate(indexOri):
        imFtrSub = imFtr.view(-1, imFtr.shape[2])
        txtFtrSub = txtFtr[idx].view(-1,1)
        simMM = torch.mm(imFtrSub, txtFtrSub)
        simMMReshape = simMM.view(-1, tube_Prp_num) 
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy()
        sort_sim_np = sortSim.data.cpu().numpy()
        frm_num = simIdx.shape[0]
        assert frm_num==len(bbx_list[0])

        tube_tmp_list = list()
        for box_id, bbx_sub_list  in enumerate(bbx_list[0]):
            tube_tmp_list.append(bbx_sub_list[simIdx[box_id, 0]])

        tube_Info_sub = tubeInfo[idx]
        tube_info_sub_prp, frm_info_list = tube_Info_sub
        prpListSort = [ [tube_tmp_list], [1]]
        shot_proposals = [prpListSort, frm_info_list]
        for ii, thre in enumerate(thre_list):
            if more_detailed_flag:
                recall_tmp, iou_list= evaluate_tube_recall_vid(shot_proposals, vid_parser, lbl, thre, topKOri=topK, more_detailed_flag=more_detailed_flag)
            else:
                recall_tmp, iou_list= evaluate_tube_recall_vid(shot_proposals, vid_parser, lbl, thre, topKOri=topK, more_detailed_flag=more_detailed_flag)
        #pdb.set_trace()
        if more_detailed_flag:
            resultList.append((lbl, recall_tmp[-1], shot_proposals, iou_list))
        else:
            resultList.append((lbl, recall_tmp[-1]))
            

        print('accuracy for %d: %3f' %(lbl, recall_tmp[-1]))

        if visFlag:
            # visualize sample results
            if(recall_tmp[-1]<=0.5):
                continue
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


def evalAcc_actNet(imFtr, txtFtr, tube_info_list, person_list, jpg_folder, visRsFd, visFlag = False, topK=1, thre_list=[0.5]):
    resultList= list()
    bSize = len(person_list)
    tube_Prp_num = imFtr.shape[1]
    stIdx = 0
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    txt_num = txtFtr.shape[1]
    for idx, person_tmp in enumerate(person_list):
        lbl = person_tmp.id
        imFtrSub = imFtr[idx]
        txtFtrSub_batch = txtFtr[idx]
        txt_num = txtFtrSub_batch.shape[0]
        for txt_idx  in range(txt_num):
            txtFtrSub = txtFtrSub_batch[txt_idx].view(-1,1)
            simMM = torch.mm(imFtrSub, txtFtrSub)
            simMMReshape = simMM.view(-1, tube_Prp_num) 
            sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
            simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
            sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)

            #print(sort_sim_np)
            tube_Info_sub = tube_info_list[idx]
            tube_info_sub_prp, frm_info_list = tube_Info_sub
            tube_info_sub_prp_bbx, tube_info_sub_prp_score = tube_info_sub_prp
            prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]] for i in range(topK)], [sort_sim_np[i] for i in range(topK)] ]
            shot_proposals = [prpListSort, frm_info_list]
            for ii, thre in enumerate(thre_list): 
                recall_tmp = evaluate_tube_recall(shot_proposals, person_tmp.shot, person_tmp, thre=thre ,topKOri=topK)
            resultList.append((lbl, recall_tmp[-1]))
            #print('accuracy for %d: %3f' %(lbl, recall_tmp[-1]))
            
            # visualize sample results
            if visFlag:
                #if(recall_tmp[-1]<0.5):
                #    continue
                
                vd_name = person_tmp.shot.video_id
                frmImNameList = [os.path.join(jpg_folder, 'v_' + vd_name, frame_name + '.png') for frame_name in frm_info_list]
                frmImList = list()
                for fId, imPath  in enumerate(frmImNameList):
                    img = cv2.imread(imPath)
                    frmImList.append(img)
                vis_frame_num = 15
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
                #pdb.set_trace()
    return resultList

def evalAcc_actNet_frm(imFtr, txtFtr, tube_info_list, person_list, jpg_folder, frm_idx_list, bbx_list, visRsFd, visFlag = False, topK=1, thre_list=[0.5]):
    resultList= list()
    bSize = len(person_list)
    tube_Prp_num = imFtr.shape[1]
    stIdx = 0
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    assert txtFtr.shape[1]==1
    for idx, person_tmp in enumerate(person_list):
        lbl = person_tmp.id
        imFtrSub = imFtr[idx]
        txtFtrSub_batch = txtFtr[idx]
        txtFtrSub = txtFtrSub_batch.view(-1,1)
        simMM = torch.mm(imFtrSub, txtFtrSub)
        simMMReshape = simMM.view(-1, tube_Prp_num) 
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
        sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)

        frm_id = frm_idx_list[idx][0]
        tmp_prp_bbx_list = bbx_list[idx][0]
        tube_Info_sub = tube_info_list[idx]
        tube_info_sub_prp, frm_info_list = tube_Info_sub
        bbx_gt_fake = person_tmp['boxes'][0]
        ov = compute_IoU_v2(tmp_prp_bbx_list[simIdx[0]], bbx_gt_fake)
        lbl = person_tmp.id
        if ov>thre_list[0]:
            resultList.append([lbl, 1])
        else:
            resultList.append([lbl, 0])

    return resultList

def evalAcc_actNet_frm_tube(imFtr, txtFtr, tube_info_list, person_list, jpg_folder, frm_idx_list, bbx_list,  visRsFd, visFlag = False, topK=1, thre_list=[0.5]):
    #pdb.set_trace()
    resultList= list()
    bSize = len(person_list)
    tube_Prp_num = imFtr.shape[1]
    stIdx = 0
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    txt_num = txtFtr.shape[1]
    assert len(person_list) ==1
    for idx, person_tmp in enumerate(person_list):
        lbl = person_tmp.id
        imFtrSub = imFtr[idx]
        txtFtrSub_batch = txtFtr[idx]
        txt_num = txtFtrSub_batch.shape[0]
        for txt_idx  in range(txt_num):
            txtFtrSub = txtFtrSub_batch[txt_idx].view(-1,1)
            simMM = torch.mm(imFtrSub, txtFtrSub)
            simMMReshape = simMM.view(-1, tube_Prp_num) 
            sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
            simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
            sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)

            #print(sort_sim_np)
            tube_Info_sub = tube_info_list[idx]
            tube_info_sub_prp, frm_info_list = tube_Info_sub
            
            tube_tmp_list = list()
            for box_id, bbx_sub_list in enumerate(bbx_list[0]):
                tube_tmp_list.append(bbx_sub_list[simIdx[box_id]])
            prpListSort = [[tube_tmp_list], [1]]
            
            frm_info_list_ann = list()
            for frm_id in frm_idx_list[0]:
                frmName =  '%05d' %(frm_id)
                frm_info_list_ann.append(frmName)
            shot_proposals = [prpListSort, frm_info_list_ann]
            for ii, thre in enumerate(thre_list): 
                recall_tmp = evaluate_tube_recall(shot_proposals, person_tmp.shot, person_tmp, thre=thre ,topKOri=topK)
            resultList.append((lbl, recall_tmp[-1]))
            print('accuracy for %d: %3f' %(lbl, recall_tmp[-1]))
            
            # visualize sample results
            if visFlag:
                if(recall_tmp[-1]<0.5):
                    continue
                
                vd_name = person_tmp.shot.video_id
                frmImNameList = [os.path.join(jpg_folder, 'v_' + vd_name, frame_name + '.png') for frame_name in frm_info_list_ann]
                frmImList = list()
                for fId, imPath  in enumerate(frmImNameList):
                    img = cv2.imread(imPath)
                    frmImList.append(img)
                vis_frame_num = 15
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
                #pdb.set_trace()
    return resultList


def evalAcc_actNet_att(simMMAtt, tube_info_list, person_list, jpg_folder, visRsFd, visFlag = False, topK=1, thre_list=[0.5]):
    # simMMRe: bSize*tube_num*bSize*capNum
    resultList= list()
    bSize = len(person_list)
    tube_Prp_num = simMMAtt.size(1)
    stIdx = 0
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    for idx, person_tmp in enumerate(person_list):
        lbl = person_tmp.id
        
        txt_num = simMMAtt.size(3)
        for txt_idx  in range(txt_num):
            simMM = simMMAtt[idx, :, idx, txt_idx]
            simMMReshape = simMM.view(-1, tube_Prp_num) 
            sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
            simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
            sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)
            #print(sort_sim_np)
            tube_Info_sub = tube_info_list[idx]
            tube_info_sub_prp, frm_info_list = tube_Info_sub
            tube_info_sub_prp_bbx, tube_info_sub_prp_score = tube_info_sub_prp
            prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]] for i in range(topK)], [sort_sim_np[i] for i in range(topK)] ]
            shot_proposals = [prpListSort, frm_info_list]
            for ii, thre in enumerate(thre_list): 
                recall_tmp = evaluate_tube_recall(shot_proposals, person_tmp.shot, person_tmp, thre=thre ,topKOri=topK)
            resultList.append((lbl, recall_tmp[-1]))
            print('accuracy for %d: %3f' %(lbl, recall_tmp[-1]))
            
            # visualize sample results
            if visFlag:
                if(recall_tmp[-1]<=0.5):
                    continue
                print(sortSim) 
                print(simIdx)
                #pdb.set_trace()
                #continue
                vd_name = person_tmp.shot.video_id
                frmImNameList = [os.path.join(jpg_folder, 'v_' + vd_name, frame_name + '.png') for frame_name in frm_info_list]
                frmImList = list()
                for fId, imPath  in enumerate(frmImNameList):
                    img = cv2.imread(imPath)
                    frmImList.append(img)
                vis_frame_num = 15
                visIner =max(int(len(frmImList) /vis_frame_num), 1)
                
                for ii in range(topK):
                    print('visualizing tube %d\n'%(ii))
                    vd_name_raw = vd_name.split('/')[-1]
                    makedirs_if_missing(visRsFd)
                    out_file_path = visRsFd+'/'+vd_name_raw+ '_' + str(ii)+'.gif'
                    if os.path.isfile(out_file_path):
                        continue
                    #pdb.set_trace() 
                    tube = prpListSort[0][ii]
                    frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
                    tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
                    tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
                    visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis_resize, out_file_path)
#                pdb.set_trace()
    return resultList


def evalAcc_actNet_ori(imFtr, txtFtr, tube_info_list, person_list, jpg_folder, visRsFd, visFlag = False, topK=1):
    resultList= list()
    bSize = len(person_list)
    tube_Prp_num = imFtr.shape[1]
    stIdx = 0
    pdb.set_trace()
    #thre_list = [0.2, 0.3, 0.4, 0.5]
    thre_list = [ 0.5]
    assert txtFtr.shape[1]==1
    for idx, person_tmp in enumerate(person_list):
        lbl = person_tmp.id
        imFtrSub = imFtr[idx]
        txtFtrSub = txtFtr[idx].view(-1,1)
        simMM = torch.mm(imFtrSub, txtFtrSub)
        simMMReshape = simMM.view(-1, tube_Prp_num) 
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy().squeeze(axis=0)
        sort_sim_np = sortSim.data.cpu().numpy().squeeze(axis=0)

        tube_Info_sub = tube_info_list[idx]
        tube_info_sub_prp, frm_info_list = tube_Info_sub
        tube_info_sub_prp_bbx, tube_info_sub_prp_score = tube_info_sub_prp
        prpListSort = [ [tube_info_sub_prp_bbx[simIdx[i]] for i in range(topK)], [sort_sim_np[i] for i in range(topK)] ]
        shot_proposals = [prpListSort, frm_info_list]
        for ii, thre in enumerate(thre_list): 
            recall_tmp = evaluate_tube_recall(shot_proposals, person_tmp.shot, person_tmp, thre=thre ,topKOri=topK)
        resultList.append((lbl, recall_tmp[-1]))
        #print('accuracy for %d: %3f' %(lbl, recall_tmp[-1]))
        
        # visualize sample results
        if visFlag:
            if(recall_tmp[-1]>=0.5):
                continue
            
            vd_name = person_tmp.shot.video_id
            frmImNameList = [os.path.join(jpg_folder, 'v_' + vd_name, frame_name + '.png') for frame_name in frm_info_list]
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

def get_upper_bound():
    opt = parse_args()
    opt.dbSet = 'actNet'
    opt.set_name = 'train'
    rpNum = 30
    tube_ftr_dim = 300
    topK = 30
    thre = 0.5

    # build dataloader
    dataLoader, datasetOri= build_dataloader(opt) 
    print(len(datasetOri))
    pdb.set_trace() 
    tube_embedding = np.zeros((1, rpNum, tube_ftr_dim), dtype=np.float32)
    txt_embedding = np.zeros((1, 1, tube_ftr_dim), dtype=np.float32)
    imFtr = torch.FloatTensor(tube_embedding).cuda()
    txtFtr = torch.FloatTensor(txt_embedding).cuda()
    
    #pdb.set_trace()
    full_result = list() 
    for index in range(len(datasetOri)):
        tube_info_index, person_index = datasetOri.get_tube_info(index)

        result_index = evalAcc_actNet(imFtr, txtFtr, [tube_info_index], [person_index], datasetOri.jpg_folder, visRsFd='../data/visResult/actNet', visFlag = False, topK=topK, thre_list=[thre])
        full_result +=result_index 
    accSum =0 
    for ele in full_result:
        index, recall_k= ele
        accSum +=recall_k
    print('Average Accuracy is %3f\n' %(accSum/len(full_result)))
    #pdb.set_trace()

def get_upper_bound_vid():
    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name = 'val'
    rpNum = 30
    tube_ftr_dim = 50
    topK = 30
    thre = 0.6

    # build dataloader
    dataLoader, datasetOri= build_dataloader(opt) 

    tube_embedding = np.zeros((1, rpNum, tube_ftr_dim), dtype=np.float32)
    txt_embedding = np.zeros((1, 1, tube_ftr_dim), dtype=np.float32)
    imFtr = torch.FloatTensor(tube_embedding).cuda()
    txtFtr = torch.FloatTensor(txt_embedding).cuda()
    
    #pdb.set_trace()
    full_result = list() 
    for indexOri in range(len(datasetOri)):
        tube_info_index, index = datasetOri.get_tube_info(indexOri)

        result_index = evalAcc(imFtr, txtFtr, [tube_info_index], [index], datasetOri, visRsFd='../data/visResult/vid', visFlag=False, topK=topK, thre_list=[thre])
        full_result +=result_index 
    accSum =0 
    for ele in full_result:
        index, recall_k= ele
        accSum +=recall_k
    print('Average Accuracy is %3f\n' %(accSum/len(full_result)))
    #pdb.set_trace()


def get_random_average_performance():
    opt = parse_args()
    opt.dbSet = 'actNet'
    opt.set_name = 'test'
    rpNum = 30
    tube_ftr_dim = 300
    topK = 1
    thre = 0.6
    # build dataloader
    dataLoader, datasetOri= build_dataloader(opt) 
    
    tube_embedding = np.zeros((1, 1, tube_ftr_dim), dtype=np.float32)
    txt_embedding = np.zeros((1, 1, tube_ftr_dim), dtype=np.float32)
    imFtr = torch.FloatTensor(tube_embedding).cuda()
    txtFtr = torch.FloatTensor(txt_embedding).cuda()
    
    full_result = list() 
    for index in range(len(datasetOri)):
        tube_info_index, person_index = datasetOri.get_tube_info(index)
        tube_prp_info, frm_list = tube_info_index
        result_index = list()
        for i in range(rpNum):
            tmp_tube_info_index = [[ [tube_prp_info[0][i]], [tube_prp_info[1][i]]] , frm_list]
            result_index += evalAcc_actNet(imFtr, txtFtr, [tmp_tube_info_index], [person_index], datasetOri.jpg_folder, visRsFd='../data/visResult/actNet_'+ str(i), visFlag = False, topK=topK, thre_list=[thre])
        acc = 0
        for ele in result_index:
            index_tmp, recall_k = ele
            acc +=recall_k
        acc_mean = acc*1.0/(len(result_index))
        result_mean = [[index_tmp, acc_mean]]
        print(result_mean)
        full_result +=result_mean 
    accSum =0 
    for ele in full_result:
        index, recall_k= ele
        accSum +=recall_k
    print('Average Accuracy is %3f\n' %(accSum/len(full_result)))


def get_random_average_performance_vid():
    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name = 'val'
    rpNum = 30
    tube_ftr_dim = 50
    topK = 1
    thre = 0.6
    # build dataloader
    dataLoader, datasetOri= build_dataloader(opt) 
    #pdb.set_trace()   

    tube_embedding = np.zeros((1, 1, tube_ftr_dim), dtype=np.float32)
    txt_embedding = np.zeros((1, 1, tube_ftr_dim), dtype=np.float32)
    imFtr = torch.FloatTensor(tube_embedding).cuda()
    txtFtr = torch.FloatTensor(txt_embedding).cuda()
    
    full_result = list() 

    for indexOri in range(len(datasetOri)):
        #pdb.set_trace()
        tube_info_index, index = datasetOri.get_tube_info(indexOri)
        tube_prp_info, frm_list = tube_info_index
        result_index = list()
        for i in range(rpNum):
            tmp_tube_info_index = [[ [tube_prp_info[0][i]], [tube_prp_info[1][i]]] , frm_list]
            result_index += evalAcc(imFtr, txtFtr, [tmp_tube_info_index], [index], datasetOri, visRsFd='../data/visResult/vid', visFlag=False, topK=topK, thre_list=[thre])
        acc = 0
        for ele in result_index:
            index_tmp, recall_k = ele
            acc +=recall_k
        acc_mean = acc*1.0/(len(result_index))
        #pdb.set_trace()
        result_mean = [[index_tmp, acc_mean]]
        print(result_mean)
        full_result +=result_mean 
    accSum =0 
    for ele in full_result:
        index, recall_k= ele
        accSum +=recall_k
    print('Average Accuracy is %3f\n' %(accSum/len(full_result)))

def get_iou_distribution_vid():
    att_result_fn ='../data/final_models/tube_result/log_bs_4_tn_30_wl_20_cn_1_fd_512_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_margin_10.0lstm_hd_512result_val_coAttV1_ep_21_lamda_1.pk'
    att_result_abs_fn ='../data/final_models/tube_result/abs__bs_1_tn_30_wl_20_cn_1_fd_512_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_margin_10.0_att_exp1lstm_hd_512result_val_coAttV1_ep_7_lamda_0.pk'
    result_att = pickleload(att_result_fn)
    result_att_abs = pickleload(att_result_abs_fn)

    result_att_tmp = result_att[4]
    result_att_abs_tmp = result_att_abs[0]
    
    sim_bin_num = 10
    sim_bin_st = [0 for i in range(sim_bin_num)]
    sim_bin_abs_st = [0 for i in range(sim_bin_num)]
    
    sim_bin_st_num = [0.00001 for i in range(sim_bin_num)]

    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name = 'val'
    rpNum = 30
    tube_ftr_dim = 50
    topK = 1
    thre = 0.5
    # build dataloader
    dataLoader, datasetOri= build_dataloader(opt) 

    tube_embedding = np.zeros((1, 1, tube_ftr_dim), dtype=np.float32)
    txt_embedding = np.zeros((1, 1, tube_ftr_dim), dtype=np.float32)
    imFtr = torch.FloatTensor(tube_embedding).cuda()
    txtFtr = torch.FloatTensor(txt_embedding).cuda()
    
    full_result = list() 

    for indexOri in range(len(datasetOri)):
        #pdb.set_trace()
        tube_info_index, index = datasetOri.get_tube_info(indexOri)
        tube_prp_info, frm_list = tube_info_index
        
        att_info = result_att_tmp[indexOri]
        att_info_abs = result_att_abs_tmp[indexOri]
        
        tmp_att_index_list = list(att_info[2])
        tmp_att_index_list_abs = list(att_info_abs[2])
        
        assert att_info[0]==att_info_abs[0]
        assert att_info[0]==index

        result_index = list()
        for i in range(rpNum):
            tmp_tube_info_index = [[ [tube_prp_info[0][i]], [tube_prp_info[1][i]]] , frm_list]
            result_index += evalAcc(imFtr, txtFtr, [tmp_tube_info_index], [index], datasetOri, visRsFd='../data/visResult/vid', visFlag=False, topK=topK, thre_list=[thre], more_detailed_flag=True)
            
            target_iou = result_index[i][-1][0]
            #pdb.set_trace()
            target_bin_id = int(math.floor(target_iou*sim_bin_num))
            sim_bin_st_num[target_bin_id] +=1

            sim_idx = tmp_att_index_list.index(i)
            sim_idx_abs = tmp_att_index_list_abs.index(i)

            sim_bin_st[target_bin_id] += att_info[3][sim_idx]
            sim_bin_abs_st[target_bin_id] += att_info_abs[3][sim_idx_abs]
        pickledump(str(sim_bin_num)+'.pk', {'sim': sim_bin_st, 'sim_st': sim_bin_abs_st, 'sim_num': sim_bin_num}) 
        acc = 0
        for ele in result_index:
            recall_k = ele[1]
            index_tmp = ele[0]
            acc +=recall_k
        acc_mean = acc*1.0/(len(result_index))
        result_mean = [[index_tmp, acc_mean]]
        print(result_mean)
        full_result +=result_mean 

    accSum =0
    min_iter = 1.0/sim_bin_num
    for i in range(sim_bin_num):
        print('%01f~%1f\n' %(min_iter*i, min_iter*(i+1)))
    for i in range(sim_bin_num):
        print('%f\n' %(sim_bin_st[i]*1.0/sim_bin_st_num[i]))
    for i in range(sim_bin_num):
        print('%3f\n' %(sim_bin_abs_st[i]*1.0/sim_bin_st_num[i]))
        #print('thre without div %3f~%3f: %3f' %(min_iter*i, min_iter*(i+1), sim_bin_abs_st[i]*1.0/sim_bin_st_num[i]))
    

    for ele in full_result:
        recall_k= ele[1]
        accSum +=recall_k
    print('Average Accuracy is %3f\n' %(accSum/len(full_result)))


def draw_bin():
    bin_data_fn='./bin_data.pk'
    sim_bin_num = 10
    
    data_bin = pickleload(bin_data_fn)
    sim_bin_st =data_bin['sim']
    sim_bin_abs_st =data_bin['sim_st']
    sim_bin_st_num = data_bin['sim_num']

    min_iter = 1.0/sim_bin_num
    for i in range(sim_bin_num):
        print('%0.1f-%0.1f' %(min_iter*i, min_iter*(i+1)))
    for i in range(sim_bin_num):
        print('%f' %(sim_bin_st[i]*1.0/sim_bin_st_num[i]))
    print('\n\n\n')
    for i in range(sim_bin_num):
        print('%f' %(sim_bin_abs_st[i]*1.0/sim_bin_st_num[i]))
        #print('thre without div %3f~%3f: %3f' %(min_iter*i, min_iter*(i+1), sim_bin_abs_st[i]*1.0/sim_bin_st_num[i]))
    


def check_vid_info():
    opt = parse_args()
    opt.dbSet = 'vid'
    opt.set_name = 'train'
    rpNum = 30
    tube_ftr_dim = 50
    topK = 30
    thre = 0.6
    lbl = 1646
    # build dataloader
    dataLoader, datasetOri= build_dataloader(opt) 
    vid_parser = datasetOri.vid_parser
    vd_name, ins_id_str = vid_parser.get_shot_info_from_index(lbl)
    ann, vd_name = vid_parser.get_shot_anno_from_index(lbl)
    pdb.set_trace()

#def evalAcc_frm_tube_test(imFtr, txtFtr, tubeInfo, indexOri, datasetOri, frm_idx_list, bbx_list, visRsFd, visFlag=False, topK = 1, thre_list=[0.5], more_detailed_flag=False):
def evalAcc_frm_tube_test():
    result_fn = '../data/final_models/tube_result/_bs_16_tn_30_wl_20_cn_1_fd_512_rankFrm_fc_none_full_txt_gru_rgb_lr_0.0_vid_margin_10.0_frm_level_result_val_rankFrm_ep_5_itr0.pk'
    result_frm = pickleload(result_fn)
    resultList= list()
    topK = 1
    thre = 0.7
    for idx in range(len(result_frm)):
        tmp_result = result_frm[idx]
        lbl = tmp_result[0]
        ov = tmp_result[3][0]
        if ov>thre:
            resultList.append((lbl, 1))
        else:
            resultList.append((lbl, 0))
        print('%d %f\n' %(lbl, ov))
    accSum = 0
    for ele in resultList:
        recall_k= ele[1]
        accSum +=recall_k
    print('Average accuracy on testing set is %3f\n' %(accSum*1.0/len(resultList)))



    return resultList

if __name__=='__main__':
    draw_bin()
    #get_iou_distribution_vid()
    #evalAcc_frm_tube_test()
    #check_vid_info() 
    #recall_K = get_upper_bound_vid()  
    #recall_K = get_upper_bound()  
    #recall_k = get_random_average_performance()
    #recall_k = get_random_average_performance_vid()
