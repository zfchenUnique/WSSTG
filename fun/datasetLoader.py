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

class a2dImDataloader(data.Dataset):
    def __init__(self, annoFile, dictFile, rpFd):
        self.data = pickleload(annoFile)
        self.dict = pickleload(dictFile)
        self.data['rpFd'] = rpFd

        #self.data['frmListGt'] = copy.deepcopy(self.data['frmList'])

        self.smpImNum= 1 
        self.rpNum = 20
        self.maxWordNum =20
        self.imW = 224
        self.imH = 224
        self.offVisFlag =True
        self.indexUsed = list()

    def image_samper_set_up(self, imNum=1, rpNum=20, capNum=1, maxWordNum=15, trainFlag=True, rndSmpImgFlag=True, usedBadWord=False,  videoWeakFlag=False, pngFd='', conSecFlag=True, conFrmNum=1):
        self.smpImNum= imNum
        self.rpNum = rpNum
        self.capNum = capNum
        self.trainFlag= trainFlag
        self.maxWordNum = maxWordNum
        self.rndSmpImgFlag = rndSmpImgFlag
        self.usedBadWord = usedBadWord
        self.conSecFlag = conSecFlag
        self.conFrmNum = conFrmNum
        setAnnotor = 0
        if not self.trainFlag:
            setAnnotor =1
        self.indexUsed = list()
        self.bbxGtIdxTest = list()  
        for i, cap in enumerate(self.data['cap']): 
            vdName = self.data['vd'][i] 
            if(self.data['splitDict'][vdName]==setAnnotor):
                if setAnnotor==1 and self.conSecFlag:
                    frmLgh = len(self.data['bbxList'][i])
                    for fId in range(frmLgh):
                        self.indexUsed.append(i)
                        self.bbxGtIdxTest.append(fId)
                else:
                    self.indexUsed.append(i)
                    
            
        #if(not rndSmpImgFlag):
        #self.data['frmList'] = copy.deepcopy(self.data['frmListGt'])

    def resize_img_rp(img, imH, imW, rpList):
        h, w, c= img.shape
        hRatio = imH/h
        wRatio = imW/w
        newRpList= list()
        for i, rp in enumerate(rpList):
            newRp  =copy.deepcopy(rp) 
            newRp[0] = rp[0]*wRatio
            newRp[2] = rp[2]*wRatio
            newRp[1] = rp[1]*hRatio
            newRp[3] = rp[3]*hRatio
            newRpList.append(newRp)
        res =cv2.resize(img, size=(imH, imW))
        return res, newRpList
         

    def pull_item_dis(self, index1):
        # word embedding
        #pdb.set_trace()
        idxUsed = self.indexUsed[index1]
        capList= self.data['cap'][idxUsed]
        wordEmbMatrix= np.zeros((self.maxWordNum, 300), dtype=np.float32)         
        valCount=0
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
        capLen = valCount+1
        #image List
        if self.rndSmpImgFlag and (not self.conSecFlag):
            subSmpIdx = random.sample(list(range(len(self.data['frmList'][idxUsed]))), self.smpImNum)
        elif self.conSecFlag and self.smpImNum!=-1:
            smpHalfL = int((self.conFrmNum-1)/2)
            subSmpSt = random.sample(list(range(smpHalfL,  len(self.data['frmList'][idxUsed])-smpHalfL)), 1)
            subSmpIdx = list(range(subSmpSt[0]-smpHalfL, smpHalfL+subSmpSt[0]+1))
        elif (not self.conSecFlag) and self.smpImNum == -1 :
            subSmpIdx = list()
            for i, frmTest in enumerate(self.data['frmListGt'][idxUsed]):
                subSmpSt = self.data['frmList'][idxUsed].index(frmTest)
                subSmpIdx.append(subSmpSt)
        elif (self.conSecFlag) and self.smpImNum == -1:
            smpHalfL = int((self.conFrmNum-1)/2)
            #pdb.set_trace()
            keyF = self.data['frmListGt'][idxUsed][self.bbxGtIdxTest[index1]]
            subSmpSt = self.data['frmList'][idxUsed].index(keyF)
            subSmpIdx = list(range(subSmpSt-smpHalfL, smpHalfL+subSmpSt+1))
        elif (self.conSecFlag) and self.smpImNum == -1 and 0:
            subSmpIdx = list()
            smpHalfL = int((self.conFrmNum-1)/2)
            for i, frmTest in enumerate(self.data['frmListGt'][idxUsed]):
                subSmpSt = self.data['frmList'][idxUsed].index(frmTest)
                subSmpIdx +=list(range(subSmpSt-smpHalfL, smpHalfL+subSmpSt+1))
        else:
            subSmpIdx = list(range(self.smpImNum))
        
        #pdb.set_trace()
        imList = list()
        rpImList = list()
        lblListVis = list()
        vdName = self.data['vd'][idxUsed] 
        lblVideo =  self.data['vd2idx'][vdName]
        for i, idx in enumerate(subSmpIdx):
#            print(idxUsed, idx)
            imgPath = self.data['rpFd']+'/' + vdName + '/' + self.data['frmList'][idxUsed][idx]+'.pd' 
            try:
                visInfo = pickleload(imgPath) 
            except:
                print(imgPath)
            rpList = visInfo['rois'][:self.rpNum] 
            imgDis = visInfo['roiFtr'][:self.rpNum]
            # to modify the rois
            newRpList = copy.deepcopy(rpList)
            imList.append(torch.from_numpy(imgDis)) 
            rpImList.append(newRpList)
            lblListVis.append(idx)
        return torch.from_numpy(wordEmbMatrix), torch.stack( imList, 0), lblVideo, lblListVis,rpImList, capLen, wordLbl 

    def pull_item_dis_test(self, index):
        wordEmb, imEmb, lblVideo, lblListVis, rpImList, capLen, wordLbl = self.pull_item_dis(index)
        # modify region proposals according to 
        vdName = self.data['idx2vd'][lblVideo]
        rpListFull= list()
        bbxGtList= list()
        #pdb.set_trace()
        idxUsed = self.indexUsed[index]
        gtIdx =0  # make sure the gt matches
        for i, idx in enumerate(lblListVis):
            frmNa = self.data['frmList'][idxUsed][idx]
            imgPath = self.data['rpFd']+'/' + vdName + '/' \
                    + self.data['frmList'][idxUsed][idx]+'.pd' 
            visInfo = pickleload(imgPath) 
            imScale = visInfo['imFo'][0, 2] 
            rpList =  rpMatPreprocess(rpImList[i], imScale, isA2D=True)
            rpListFull.append(rpList)
            #pdb.set_trace()
            if frmNa not in self.data['frmListGt'][idxUsed]:
                continue
            if i%self.conFrmNum !=(self.conFrmNum-1)/2: # only return gt for gt Frame
                continue
            if self.smpImNum==-1 and (self.conSecFlag):
                bbxIdxUsed = self.bbxGtIdxTest[index]
                bbxGt=copy.deepcopy(list(self.data['bbxList'][idxUsed][bbxIdxUsed]))
            elif self.smpImNum==-1 and (not self.conSecFlag):
                bbxGt=copy.deepcopy(list(self.data['bbxList'][idxUsed][gtIdx]))
                gtIdx +=1
            else:
                boxIdx = self.data['frmListGt'][idxUsed].index(frmNa)
                bbxGt=copy.deepcopy(list(self.data['bbxList'][idxUsed][boxIdx]))
            #gtIdx +=1
            bbxGt[2] = bbxGt[2] -bbxGt[0]
            bbxGt[3] = bbxGt[3] -bbxGt[1]
            bbxGtList.append(bbxGt)
        return wordEmb, imEmb, lblVideo, lblListVis, rpListFull, capLen, bbxGtList, idxUsed, wordLbl

    def __len__(self):
        return len(self.indexUsed)

    def __getitem__(self, index):
        if not self.offVisFlag and self.trainFlag and 0:
            wordEmd, img, proposals = self.pull_item_vis(index)
        elif self.offVisFlag and self.rndSmpImgFlag and 0:
            wordEmbMatrix, img, lbl, frmList, proposals, capLen, wordLbl= self.pull_item_dis(index)
            return img,  wordEmbMatrix, lbl, capLen, proposals, wordLbl
        else:
            wordEmbMatrix, img, lbl, frmList, proposals, capLen, bbxGtList, capLbl, wordLbl= self.pull_item_dis_test(index)
            return img,  wordEmbMatrix, lbl, capLen, proposals, bbxGtList, frmList, capLbl, wordLbl

    #  load proposals for all the frames in the videos, for extracting tubes
    def load_video_proposals(self, index):
        vdName = self.data['vd'][index]
        vdIdx = self.data['vd2idx'][vdName]
        frmLgh = len(self.data['frmList'][vdIdx])
        det_prp_list = list()
        for idx in range(frmLgh):
            imgPath = self.data['rpFd']+'/' + vdName + '/' + self.data['frmList'][vdIdx][idx]+'.pd' 
            try:
                visInfo = pickleload(imgPath) 
            except:
                print(imgPath)
            prpNum = self.rpNum
            if visInfo['rois'].shape[0]<prpNum:
                prpNum = visInfo['rois'].shape[0]
            rpList = visInfo['rois'][:prpNum] 
            imScale = visInfo['imFo'][0, 2] 
            rpList =  rpMatPreprocess(rpList, imScale, isA2D=True)
            rpListMat = np.vstack(rpList)
            rpListMat[:, 2] = rpListMat[:, 2] + rpListMat[:, 0]
            rpListMat[:, 3] = rpListMat[:, 3] + rpListMat[:, 1]
            rpScoreList = np.expand_dims(visInfo['roisS'][:prpNum], axis=1 )
            det_prp_list.append([rpScoreList, rpListMat])
        pdb.set_trace()
        return det_prp_list

def dis_collate(batch):
    targets = []
    imgs = []
    text = []
    rprList = []
    maxLen = batch[0][3]
    gtBbxList = []
    frmList= []
    lblCap = []
    wordLbl = []
    capLgList = []
    for sample in batch:
        imgs.append(sample[0])
        text.append(sample[1])
        targets.append(sample[2])
        if(sample[3]>maxLen):
            maxLen = sample[3]
        rprList.append(sample[4])
        capLgList.append(sample[3])
        if(len(sample)>6):
            gtBbxList.append(sample[5])
            frmList.append(sample[6])
            lblCap.append(sample[7])
            wordLbl.append(sample[8])
        else:
            wordLbl.append(sample[5])
    # shorten the word EMb for faster training
    capMatrix = torch.stack(text, 0)
    capMatrix = capMatrix[:, :maxLen, :]
    if len(batch[0])>6:
        return torch.cat(imgs, 0), capMatrix, targets, rprList, gtBbxList, frmList, lblCap, wordLbl, capLgList 
    return torch.stack(imgs, 0), capMatrix, targets, wordLbl, capLgList 

def build_dataloader(opt):
    if opt.dbSet=='otb':
        dataset = otbImDataloader(inFd='/disk2/zfchen/data/OTB_sentences',
            annoFile='../data/annForDb_otbV2.pd',
            dictFile='../data/dictForDb_otb.pd',
            rpFd='/disk2/zfchen/data/otbRpn')
        dataset.image_samper_set_up(rpNum=opt.k_prp, imNum=opt.k_img, \
                maxWordNum=opt.maxWL,trainFlag=True)
    elif opt.dbSet=='a2d':
        annoFile = '../data/annoted_a2dV2.pd'
        dictFile = '../data/dictForDb_a2d.pd'
        #rpFd ='/disk2/zfchen/data/a2dRP'
        rpFd ='/data1/zfchen/data/a2dRP'
        vdFrmFd ='/data1/zfchen/data/A2D/Release/pngs320H'
        dataset = a2dImDataloader(annoFile, dictFile, rpFd)
        dataset.image_samper_set_up(rpNum=opt.k_prp, imNum=opt.k_img, \
                maxWordNum=opt.maxWL,trainFlag=True, videoWeakFlag=opt.vwFlag, pngFd=vdFrmFd, \
                conSecFlag=opt.conSecFlag, conFrmNum=opt.conFrmNum

                )
    else:
        print('Not implemented for dataset %s\n' %(opt.dbSet))
        return
     
    data_loader = data.DataLoader(dataset,  opt.batchSize, \
            num_workers=opt.num_workers, collate_fn=dis_collate, \
            shuffle=True, pin_memory=True)
    return data_loader, dataset

if __name__=='__main__':
    opt = parse_args()
    annoFile = '../data/annoted_a2dV2.pd'
    dictFile = '../data/dictForDb_a2d.pd'
    #rpFd ='/disk2/zfchen/data/a2dRP'
    rpFd ='/data1/zfchen/data/a2dRP'
    vdFrmFd ='/data1/zfchen/data/A2D/Release/pngs320H'
    dataset = a2dImDataloader(annoFile, dictFile, rpFd)
    dataset.image_samper_set_up(rpNum=opt.k_prp, imNum=opt.k_img, \
            maxWordNum=opt.maxWL,trainFlag=True, videoWeakFlag=opt.vwFlag, pngFd=vdFrmFd, \
            conSecFlag=opt.conSecFlag, conFrmNum=opt.conFrmNum

            )
    vId = 0
    det_list= dataset.load_video_proposals(vId)
    results = get_tubes(det_list, 0.1)
    frmImList = list()
    for fId, fName  in enumerate(dataset.data['frmList'][vId]):
        vdName = dataset.data['vd'][vId]
        imPath =  vdFrmFd +  '/' + vdName +'/' + fName + '.png'
        img = cv2.imread(imPath)
        frmImList.append(img)
    pdb.set_trace()
    for i in range(len(results[0])):
   
        tube = results[0][i]
        visTube_from_image(copy.deepcopy(frmImList), tube, 'sample/' + str(i)+'.gif')

    

    print('finsh testing')

