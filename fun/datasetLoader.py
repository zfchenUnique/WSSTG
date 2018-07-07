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

    def image_samper_set_up(self, imNum=1, rpNum=20, capNum=1, maxWordNum=15, trainFlag=True):
        self.smpImNum= imNum
        self.rpNum = rpNum
        self.capNum = capNum
        self.trainFlag= trainFlag
        self.maxWordNum = maxWordNum
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
        for i, word in enumerate(capList):
            if(i>self.maxWordNum):
                break
            idx = self.dict['word2idx'][word]
            wordEmbMatrix[i, :]= self.dict['word2vec'][idx]

        #image List
        subSmpIdx = random.sample(list(range(len(self.gtBbxDict[index]))), self.smpImNum)
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
            newRpList = rpList
            imList.append(torch.from_numpy(imgDis)) 
            rpImList.append(newRpList)
            lblListVis.append(idx)
        return torch.from_numpy(wordEmbMatrix), torch.stack( imList, 0), lblTxt, lblListVis,rpImList 

    def __len__(self):
        return len(self.vdNameList)

    def __getitem__(self, index):
        if not self.offVisFlag:
            wordEmd, img, proposals = self.pull_item_vis(index)
        else:
            wordEmbMatrix, img, lbl, frmList, proposals= self.pull_item_dis(index)
        return img,  wordEmbMatrix, lbl

    
def dis_collate(batch):
    targets = []
    imgs = []
    text = []
    for sample in batch:
        imgs.append(sample[0])
        text.append(sample[1])
        targets.append(sample[2])
    
    return torch.stack(imgs, 0), torch.stack(text, 0),targets

def build_dataloader(opt):
    if opt.dbSet=='otb':
        dataset = otbImDataloader(inFd='/disk2/zfchen/data/OTB_sentences',
            annoFile='../data/annForDb_otb.pd',
            dictFile='../data/dictForDb_otb.pd',
            rpFd='/disk2/zfchen/data/otbRpn')
        dataset.image_samper_set_up(rpNum=opt.k_prp, imNum=opt.k_img, \
                maxWordNum=opt.maxWL,trainFlag=True)
    else:
        print('Not implemented for dataset %s\n' %(opt.dbSet))
        return
     
    data_loader = data.DataLoader(dataset,  opt.batchSize, \
            num_workers=opt.num_workers, collate_fn=dis_collate, pin_memory=True)
    return data_loader



if __name__=='__main__':
    otbTester = otbImDataloader(inFd='/disk2/zfchen/data/OTB_sentences',
       annoFile='../data/annForDb_otb.pd',
      dictFile='../data/dictForDb_otb.pd',
      rpFd='/disk2/zfchen/data/otbRpn')
    otbTester.image_samper_set_up(trainFlag=False)
    img, embed, lbl= otbTester.__getitem__(0)
    print('finsh testing')

