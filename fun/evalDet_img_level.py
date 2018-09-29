import os
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


def evalAcc(imFtr, txtFtr, lblList, prpList, gtBbxList, vidNameList, frmList, capLbl, annDict=None, fdPre = '../data/A2DPrpGraphRankTest' ):
    resultList= list()
    bSize = len(lblList)
    prpListOri= prpList[0][0]
    kPrp = len(prpListOri)
    frmNum = len(frmList[0])
    imFtr = imFtr.view(-1, kPrp, imFtr.shape[2])
    txtFtr = txtFtr.squeeze(0)
    topK= 1

    stIdx = 0
    for idx, lbl in enumerate(capLbl):
        vdName = vidNameList[lbl]
        endIdx = stIdx +len(frmList[idx])
        imFtrSub = imFtr[stIdx:endIdx, :, :]
        imFtrSub = imFtrSub.view(-1, imFtrSub.shape[2])
        stIdx = endIdx
        txtFtrSub = txtFtr[idx, :].unsqueeze(1)
        simMM = torch.mm(imFtrSub, txtFtrSub)
        # evaluate each frames
        #pdb.set_trace()                
        simMMReshape = simMM.view(-1, kPrp) 
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy()
        prpListSortMf = list() 
        for fId in range(sortSim.shape[0]):
            prpListOri= copy.deepcopy(prpList[idx][fId])
            prpListSort = [prpListOri[simIdx[fId, i]] for i in range(kPrp)]
            prpListSortMf.append(prpListSort)

            # visualize sample results
            if annDict is not None:
                #imFullPath  = annDict['vFd'] + '/' + vdName +'/img/' \
                # annDict['trainImg'][lbl][fId] +'.jpg'
                #imFullPath  = '/disk2/zfchen/data/A2D/Release/pngs320H/'  \
                imId = frmList[idx][fId]
                imFullPath  = '/data1/zfchen/data/A2D/Release/pngs320H/'  \
                        + vdName +'/'+ annDict['frmList'][lbl][imId] +'.png'
                img = cv2.imread(imFullPath)
                img2 = copy.deepcopy(img)
                imSize =  np.array(get_image_size(imFullPath))
                rpScore = sortSim[fId,:topK].cpu().data.numpy()
                rpScore = np.expand_dims(rpScore, axis=1)
                #rpM = np.array(prpListSortMf) 
                rpM = np.array(prpListSort)[:topK, :] 
                #print(rpM.shape)
                #pdb.set_trace()
                bbxInfo = np.concatenate((rpM, rpScore), 1) 
                bbxInfo[:, 2]= bbxInfo[:, 2] + bbxInfo[:, 0]
                bbxInfo[:, 3]= bbxInfo[:, 3] + bbxInfo[:, 1]
                #print(bbxInfo)
                im_show = vis_detections(img, bbxInfo, thresh=-1, topK=topK, color=0)
                bbxGtIJ = copy.deepcopy(gtBbxList[idx][fId])
                #bbxGtIJ = gtBbxList[idx][fId]
                bbxGtIJ[2] = bbxGtIJ[2]+bbxGtIJ[0]
                bbxGtIJ[3] = bbxGtIJ[3]+bbxGtIJ[1]
                bbxGtIJ.append(1)
                gtArr = np.array(bbxGtIJ)
                gtArr = np.expand_dims(gtArr, axis=0)
                im_show2 = vis_detections(img2, gtArr, thresh=-1, topK=1, color=1)
                im_show2= putCapOnImage(im_show2, annDict['cap'][lbl])

                imNameRaw= os.path.basename(imFullPath).split('.')[0]
                makedirs_if_missing(fdPre)
                detName = fdPre + '/'+ vdName +'_' + imNameRaw +'_'+str(imId)+'_det.jpg'
                gtName = fdPre + '/'+ vdName +'_'+ imNameRaw +'_'+str(imId)+'_gt.jpg'

                cv2.imwrite(gtName, im_show2)
                cv2.imwrite(detName, im_show)

        evalAccObj = evalDetAcc(gtBbxList[idx], IoU=0.5, topK=topK)
        acc = evalAccObj.evalList(prpListSortMf) 
        print('accuracy for %s: %3f' %(vdName, acc))
        resultList.append((vdName, acc, frmList[idx]))
    return resultList

def evalAccGroundR(rpSS, logMat, lblList, prpList, gtBbxList, vidNameList, frmList, capLbl, annDict=None):
    resultList= list()
    bSize = len(lblList)
    prpListOri= prpList[0][0]
    kPrp = len(prpListOri)
    frmNum = len(frmList[0])
    rpSS = rpSS.view(-1, kPrp)
    topK= 1
    stIdx = 0
    
    for idx, lbl in enumerate(capLbl):
        vdName = vidNameList[lbl]
        endIdx = stIdx +len(frmList[idx])
        simMM = rpSS[stIdx:endIdx, :]
        stIdx = endIdx
        # evaluate each frames
        #pdb.set_trace()                
        simMMReshape = simMM.view(-1, kPrp) 
        sortSim, simIdx = torch.sort(simMMReshape, dim=1, descending=True)
        simIdx = simIdx.data.cpu().numpy()
        prpListSortMf = list() 
        for fId in range(sortSim.shape[0]):
            prpListOri= copy.deepcopy(prpList[idx][fId])
            prpListSort = [prpListOri[simIdx[fId, i]] for i in range(kPrp)]
            prpListSortMf.append(prpListSort)

            # visualize sample results
            if annDict is not None:
                #imFullPath  = annDict['vFd'] + '/' + vdName +'/img/' \
                # annDict['trainImg'][lbl][fId] +'.jpg'
                #imFullPath  = '/disk2/zfchen/data/A2D/Release/pngs320H/'  \
                imFullPath  = '/data1/zfchen/data/A2D/Release/pngs320H/'  \
                        + vdName +'/'+ annDict['frmList'][lbl][fId] +'.png'
                img = cv2.imread(imFullPath)
                img2 = copy.deepcopy(img)
                imSize =  np.array(get_image_size(imFullPath))
                rpScore = sortSim[fId,:topK].cpu().data.numpy()
                rpScore = np.expand_dims(rpScore, axis=1)
                #rpM = np.array(prpListSortMf) 
                rpM = np.array(prpListSort)[:topK, :] 
                #print(rpM.shape)
                #pdb.set_trace()
                bbxInfo = np.concatenate((rpM, rpScore), 1) 
                bbxInfo[:, 2]= bbxInfo[:, 2] + bbxInfo[:, 0]
                bbxInfo[:, 3]= bbxInfo[:, 3] + bbxInfo[:, 1]
                #print(bbxInfo)
                im_show = vis_detections(img, bbxInfo, thresh=-1, topK=topK, color=0)
                bbxGtIJ = copy.deepcopy(gtBbxList[idx][fId])
                #bbxGtIJ = gtBbxList[idx][fId]
                bbxGtIJ = list(bbxGtIJ)
                bbxGtIJ[2] = bbxGtIJ[2]+bbxGtIJ[0]
                bbxGtIJ[3] = bbxGtIJ[3]+bbxGtIJ[1]
                bbxGtIJ.append(1)
                gtArr = np.array(bbxGtIJ)
                gtArr = np.expand_dims(gtArr, axis=0)
                im_show2 = vis_detections(img2, gtArr, thresh=-1, topK=1, color=1)
        
                imNameRaw= os.path.basename(imFullPath).split('.')[0]
                fdPre = '../data/A2DPrpGr'
                makedirs_if_missing(fdPre)
                detName = fdPre + '/'+ vdName +'_' + imNameRaw +'_'+str(fId)+'_det.jpg'
                gtName = fdPre + '/'+ vdName +'_'+ imNameRaw +'_'+str(fId)+'_gt.jpg'

                cv2.imwrite(gtName, im_show2)
                cv2.imwrite(detName, im_show)

        evalAccObj = evalDetAcc(gtBbxList[idx], IoU=0.5, topK=topK)
        acc = evalAccObj.evalList(prpListSortMf) 
        print('accuracy for %s: %3f' %(vdName, acc))
        resultList.append((vdName, acc, frmList[idx]))
    return resultList





def testOTb():
    topK = 20 
    dbAnn = '../data/annForDb_otbV2.pd'
    dbSetFd = '/disk2/zfchen/data/OTB_sentences/OTB_videos'
    prpPth ='/disk2/zfchen/data/otbRpn' 
    
    otbDbANNDict= pickleload(dbAnn) 
    #vidNameList =otbDbANNDict['testName']
    #bbxList = otbDbANNDict['test_bbx_list']
    #frmList = otbDbANNDict['testImg']
    vidNameList =otbDbANNDict['trainName']
    bbxList = otbDbANNDict['train_bbx_list']
    frmList = otbDbANNDict['trainImg']
    vis_flag= True
    
    for i, vidName in enumerate(vidNameList):
        #if(vidName!='KiteSurf'):
        #    continue


        bbxGtI = bbxList[i]
        frmListI = frmList[i]
        evalAccObj = evalDetAcc(bbxGtI, IoU=0.5, topK=topK)
        rpListVd = list()
        
        print(i, vidName )
        for j, bbx in enumerate(bbxGtI):
            #imName= str(j+1).zfill(4)
            imName = frmListI[j] 
            imFullPath = dbSetFd + '/' + vidName + \
                    '/img/' + imName +'.jpg'
            rpFullPath = prpPth + '/' + vidName + \
                    '/' + imName +'.pd'
            #print(rpFullPath)
            try:
                rpInfo = cPickleload(rpFullPath)
            except:
                print('bad frame %s\n' %(rpFullPath))
            rpMatrix = rpInfo['rois'][:topK, :]

            #pdb.set_trace() 
            if vis_flag:
                img = cv2.imread(imFullPath)
                img2 = copy.deepcopy(img)
                rpScore = rpInfo['roisS'][:topK]
                rpScore = np.expand_dims(rpScore, axis=1)
                imSize =  np.array(get_image_size(imFullPath))
                #pdb.set_trace() 
                rpList =  rpMatPreprocess(rpMatrix, imSize)
                rpM = np.array(rpList) 
                #print(rpM.shape)
                bbxInfo = np.concatenate((rpM, rpScore), 1) 
                #print(bbxInfo)
                bbxInfo[:, 2]= bbxInfo[:, 2] + bbxInfo[:, 0]
                bbxInfo[:, 3]= bbxInfo[:, 3] + bbxInfo[:, 1]
                im_show = vis_detections(img, bbxInfo, thresh=-1, topK=5, color=0)
                bbxGtIJ = bbxGtI[j]
                bbxGtIJ[2] = bbxGtIJ[2]+bbxGtIJ[0]
                bbxGtIJ[3] = bbxGtIJ[3]+bbxGtIJ[1]
                bbxGtIJ.append(1)
                gtArr = np.array(bbxGtIJ)
                gtArr = np.expand_dims(gtArr, axis=0)
                im_show2 = vis_detections(img2, gtArr, thresh=-1, topK=20, color=1)
          
                cv2.imwrite('test_gt.jpg', im_show2)
                cv2.imwrite('test_prp.jpg', im_show)

                pdb.set_trace() 

            imSize =  np.array(get_image_size(imFullPath))
            rpList =  rpMatPreprocess(rpMatrix, imSize)
            rpListVd.append(rpList)
        acc = evalAccObj.evalList(rpListVd) 
        print('accuracy for vid: %3f' %(acc))

def keepKeyFrmForTest(imFtrM,  prpList, frmList):
    bSize = len(prpList)
    frmSize = len(prpList[0])
    prpSize = len(prpList[0][0])
    frmIdx = (frmSize-1)/2
    imFtrM = imFtrM[:, frmIdx, :, :]
    imFtrM = imFtrM.contiguous()

    prpListNew = list()
    frmListNew = list()
    #pdb.set_trace()
    for i in range(bSize):
        prpListNew.append([prpList[i][frmIdx]])
        frmListNew.append([frmList[i][frmIdx]]) 
    return imFtrM, prpListNew, frmListNew

def drawGt():
    annoFile = '../data/annoted_a2d.pd'
    dictFile = '../data/dictForDb_a2d.pd'
    #rpFd ='/disk2/zfchen/data/a2dRP'
    rpFd ='/data1/zfchen/data/a2dRP'
    vdFrmFd ='/data1/zfchen/data/A2D/Release/pngs320H'
    dataset = a2dImDataloader(annoFile, dictFile, rpFd)
    dataset.image_samper_set_up(rpNum=20, imNum=1, \
        maxWordNum=20, trainFlag=True, videoWeakFlag=False, pngFd=vdFrmFd)
    dataset.data['pngFd'] = vdFrmFd
    
    for i, icapSeg in enumerate(dataset.data['cap']):
        vdName = dataset.data['vd'][i] 
        subImFullPath = dataset.data['pngFd'] +'/' + vdName 
        subFrmList = dataset.data['frmList'][i]
        subBbxList = dataset.data['bbxList'][i]
        if(dataset.data['splitDict'][vdName]==0):
            continue
        for j, frmName in enumerate(subFrmList):
            bbxGtIJ = subBbxList[j]
            imFullPath = subImFullPath + '/' + frmName +'.png'
            img2 = cv2.imread(imFullPath)
            bbxGtIJ = list(bbxGtIJ)
            #bbxGtIJ[2] = bbxGtIJ[2]+bbxGtIJ[0]
            #bbxGtIJ[3] = bbxGtIJ[3]+bbxGtIJ[1]
            bbxGtIJ.append(1)
            gtArr = np.array(bbxGtIJ)
            gtArr = np.expand_dims(gtArr, axis=0)
            im_show2 = vis_detections(img2, gtArr, thresh=-1, topK=1, color=1)
            gtName = '../data/A2DPrpGt2/'+ vdName +'_'+ frmName +'_'+str(j)+'_gt.jpg'
            cv2.imwrite(gtName, im_show2)
    print('finish')

if __name__=='__main__':
    drawGt()






                 
