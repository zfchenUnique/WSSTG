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

class otbImDataloader(data.Dataset):
    def __init__(self, inFd, annoFile, dictFile, rpFd):
        self.data = pickleload(annoFile)
        self.dict = pickleload(dictFile)
        self.data['qtsFd'] = inFd + '/OTB_query_test'     
        self.data['qtrFd'] = inFd + '/OTB_query_train'     
        self.data['vFd'] = inFd + '/OTB_videos'
        self.data['rpFd'] = rpFd

        self.smpImNum= 1 
        self.rpNum = 20
        self.maxWordNum =15
        self.imW = 224
        self.imH = 224
        self.offVisFlag =True

    def image_samper_set_up(self, imNum=1, rpNum=20, capNum=1, maxWordNum=15, trainFlag=True, rndSmpImgFlag=True, videoWeakFlag=False):
        self.smpImNum= imNum
        self.rpNum = rpNum
        self.capNum = capNum
        self.trainFlag= trainFlag
        self.maxWordNum = maxWordNum
        self.rndSmpImgFlag = rndSmpImgFlag
        if(self.trainFlag):
            self.capList=self.data['trainCap']
            self.vdNameList=self.data['trainName']
            self.frameDict =self.data['trainImg']
            self.gtBbxDict =self.data['train_bbx_list']
        else:
            self.capList = self.data['testCap']
            self.vdNameList= self.data['testName']
            self.frameDict = self.data['testImg']
            self.gtBbxDict = self.data['test_bbx_list']
        if(videoWeakFlag):
            self.data['testImg'] = extAllFrmFn(self.vdNameList, self.data['vFd'])

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


    def pull_item_vis(self, index):
        # word embedding
        capList= self.capList[index][0]
        wordEmbMatrix= np.zeros((self.maxWordNum, 300), dtype=np.float32)         
        for i, word in enumerate(capList):
            if(i>self.maxWordNum):
                break
            idx = self.dict['word2idx'][word]
            wordEmbMatrix[i, :]= self.dict['word2vec'][idx]

        #image List
        subSmpIdx = random.sample(list(range(len(self.gtBbxDict[index]))), self.smpImNum)
        imList = list()
        rpImList = list()
        for i, idx in enumerate(subSmpIdx):
            imgPath = self.data['vFd']+'/'+self.vdNameList[index] +'/img/'+self.frameDict[index][idx]+'.jpg' 
            img = cv2.imread(imgPath) 
            rpFn = self.data['rpFd']+'/'+self.vdNameList[index]+ '/img/' +self.frameDict[index][idx]+'.mat'
            rpList = sio.loadmat(rpFn)
            imgRe, newRpList = resize_img_rp(img, rpList) 
            imList.append(torch.from_numpy(imgRe).permute(2, 0, 1)) 
            rpImList.append(newRpList)
        return wordEmbMatrix, torch.stack( imList, 0), rpImList 

    def pull_item_dis(self, index):
        # word embedding
        capList= self.capList[index][0]
        wordEmbMatrix= np.zeros((self.maxWordNum, 300), dtype=np.float32)         
        valCount=0
        for i, word in enumerate(capList):
            if word in self.dict['out_voca']:
                #print('Invalid word: ',  idx, word, len(self.dict['word2vec']))
                continue
            if(valCount>self.maxWordNum):
                break
            idx = self.dict['word2idx'][word]
            wordEmbMatrix[valCount, :]= self.dict['word2vec'][idx]
            valCount +=1
        capLen = valCount+1
        #image List
        if self.rndSmpImgFlag:
            subSmpIdx = random.sample(list(range(len(self.gtBbxDict[index]))), self.smpImNum)
        else:
            subSmpIdx = list(range(self.smpImNum))
        
        imList = list()
        rpImList = list()
        lblTxt= index
        lblListVis = list()
        vdName = self.vdNameList[index] 
        for i, idx in enumerate(subSmpIdx):
            imgPath = self.data['rpFd']+'/' + vdName + '/' + self.frameDict[index][idx]+'.pd' 
            visInfo = pickleload(imgPath) 
            rpList = visInfo['rois'][:self.rpNum] 
            imgDis = visInfo['roiFtr'][:self.rpNum]
            # to modify the rois
            newRpList = copy.deepcopy(rpList)
            imList.append(torch.from_numpy(imgDis)) 
            rpImList.append(newRpList)
            lblListVis.append(idx)
        return torch.from_numpy(wordEmbMatrix), torch.stack( imList, 0), lblTxt, lblListVis,rpImList, capLen 


    def pull_item_dis_test(self, index):
        wordEmb, imEmb, lblTxt, lblListVis, rpImList, capLen = self.pull_item_dis(index)
        # modify region proposals according to 
        vdName = self.vdNameList[lblTxt]
        rpListFull= list()
        bbxGtList= list()
        #pdb.set_trace()
        for i, idx in enumerate(lblListVis):
            imFullPath = self.data['vFd']+'/'+vdName + '/img/'+ \
                    self.frameDict[lblTxt][idx]+'.jpg'
            imSize =  np.array(get_image_size(imFullPath))
            rpList =  rpMatPreprocess(rpImList[i], imSize)
            rpListFull.append(rpList)
            bbxGtList.append(self.gtBbxDict[lblTxt][idx])
        
        return wordEmb, imEmb, lblTxt, lblListVis, rpListFull, capLen, bbxGtList


    def __len__(self):
        return len(self.vdNameList)

    def __getitem__(self, index):
        if not self.offVisFlag and self.trainFlag:
            wordEmd, img, proposals = self.pull_item_vis(index)
        elif self.offVisFlag and self.rndSmpImgFlag:
            wordEmbMatrix, img, lbl, frmList, proposals, capLen= self.pull_item_dis(index)
            return img,  wordEmbMatrix, lbl, capLen, proposals
        else:
            wordEmbMatrix, img, lbl, frmList, proposals, capLen, bbxGtList= self.pull_item_dis_test(index)
            return img,  wordEmbMatrix, lbl, capLen, proposals, bbxGtList, frmList


class a2dImDataloader(data.Dataset):
    def __init__(self, annoFile, dictFile, rpFd):
        self.data = pickleload(annoFile)
        self.dict = pickleload(dictFile)
        self.data['rpFd'] = rpFd

        self.data['frmListGt'] = copy.deepcopy(self.data['frmList'])

        self.smpImNum= 1 
        self.rpNum = 20
        self.maxWordNum =20
        self.imW = 224
        self.imH = 224
        self.offVisFlag =True
        self.indexUsed = list()

    def image_samper_set_up(self, imNum=1, rpNum=20, capNum=1, maxWordNum=15, trainFlag=True, rndSmpImgFlag=True, usedBadWord=False,  videoWeakFlag=False, pngFd='', conSecFlag=True):
        self.smpImNum= imNum
        self.rpNum = rpNum
        self.capNum = capNum
        self.trainFlag= trainFlag
        self.maxWordNum = maxWordNum
        self.rndSmpImgFlag = rndSmpImgFlag
        self.usedBadWord = usedBadWord
        self.conSecFlag = conSecFlag
        setAnnotor = 0
        if not self.trainFlag:
            setAnnotor =1
        self.indexUsed = list()
        for i, cap in enumerate(self.data['cap']): 
            vdName = self.data['vd'][i] 
            if(self.data['splitDict'][vdName]==setAnnotor):
                self.indexUsed.append(i)
        if(videoWeakFlag or conSecFlag):
            self.data['frmList'] = extAllFrmFn(self.data['vd'], pngFd)
        else:
            self.data['frmList'] = copy.deepcopy(self.data['frmListGt'])


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
        elif self.conSecFlag and self.smpImNum>1:
            smpHalfL = int((self.smpImNum-1)/2)
            subSmpSt = random.sample(list(range(smpHalfL,  len(self.data['frmList'][idxUsed])-smpHalfL)), 1)
            subSmpIdx = list(range(subSmpSt[0]-smpHalfL, smpHalfL+subSmpSt[0]+1))
        elif (not self.conSecFlag) and self.smpImNum == -1:
            subSmpIdx = list(range(len(self.data['bbxList'][idxUsed])))
        elif (self.conSecFlag) and self.smpImNum == -1:
            subSmpIdx = list()
            smpHalfL = int((self.smpImNum-1)/2)
            for i, frmTest in enumerate(self.data['frmList']):
                subSmpSt = self.data['frmList'].index(frmTest)
                subSmpIdx +=list(range(subSmpSt[0]-smpHalfL, smpHalfL+subSmpSt[0]+1))
        else:
            subSmpIdx = list(range(self.smpImNum))
        
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
#                pdb.set_trace()
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
        for i, idx in enumerate(lblListVis):
            frmNa = self.data['frmList'][idxUsed][idx]
            if frmNa not in self.data['frmList']:
                continue
            imgPath = self.data['rpFd']+'/' + vdName + '/' \
                    + self.data['frmList'][idxUsed][idx]+'.pd' 
            visInfo = pickleload(imgPath) 
            imScale = visInfo['imFo'][0, 2] 
            rpList =  rpMatPreprocess(rpImList[i], imScale, isA2D=True)
            rpListFull.append(rpList)
            bbxGt=copy.deepcopy(list(self.data['bbxList'][idxUsed][idx]))
            bbxGt[2] = bbxGt[2] -bbxGt[0]
            bbxGt[3] = bbxGt[3] -bbxGt[1]
            bbxGtList.append(bbxGt)
        return wordEmb, imEmb, lblVideo, lblListVis, rpListFull, capLen, bbxGtList, idxUsed, wordLbl

    def __len__(self):
        return len(self.indexUsed)

    def __getitem__(self, index):
        if not self.offVisFlag and self.trainFlag:
            wordEmd, img, proposals = self.pull_item_vis(index)
        elif self.offVisFlag and self.rndSmpImgFlag:
            wordEmbMatrix, img, lbl, frmList, proposals, capLen, wordLbl= self.pull_item_dis(index)
            return img,  wordEmbMatrix, lbl, capLen, proposals, wordLbl
        else:
            wordEmbMatrix, img, lbl, frmList, proposals, capLen, bbxGtList, capLbl, wordLbl= self.pull_item_dis_test(index)
            return img,  wordEmbMatrix, lbl, capLen, proposals, bbxGtList, frmList, capLbl, wordLbl


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
        annoFile = '../data/annoted_a2d.pd'
        dictFile = '../data/dictForDb_a2d.pd'
        #rpFd ='/disk2/zfchen/data/a2dRP'
        rpFd ='/data1/zfchen/data/a2dRP'
        vdFrmFd ='/data1/zfchen/data/A2D/Release/pngs320H'
        dataset = a2dImDataloader(annoFile, dictFile, rpFd)
        dataset.image_samper_set_up(rpNum=opt.k_prp, imNum=opt.k_img, \
                maxWordNum=opt.maxWL,trainFlag=True, videoWeakFlag=opt.vwFlag, pngFd=vdFrmFd, \
                conSecFlag=opt.conSecFlag

                )
    else:
        print('Not implemented for dataset %s\n' %(opt.dbSet))
        return
     
    data_loader = data.DataLoader(dataset,  opt.batchSize, \
            num_workers=opt.num_workers, collate_fn=dis_collate, \
            shuffle=True, pin_memory=True)
    return data_loader, dataset


    

if __name__=='__main__':
    otbTester = otbImDataloader(inFd='/disk2/zfchen/data/OTB_sentences',
       annoFile='../data/annForDb_otb.pd',
      dictFile='../data/dictForDb_otb.pd',
      rpFd='/disk2/zfchen/data/otbRpn')
    otbTester.image_samper_set_up(trainFlag=False)
    img, embed, lbl= otbTester.__getitem__(0)
    print('finsh testing')

