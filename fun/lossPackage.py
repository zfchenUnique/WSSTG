import torch 
import torch.nn as nn
import torch.nn.functional as F
import pdb

class lossEvaluator(nn.Module):
    def __init__(self, margin=0.1, biLossFlag=True, lossWFlag=False, lamda=0.8, keepKeyFrameOnly=False):
        super(lossEvaluator, self).__init__()
        self.margin =margin
        self.biLossFlag = biLossFlag
        self.lossWFlag = lossWFlag
        self.lamda= lamda
        self.keepKeyFrameOnly = keepKeyFrameOnly

    def forward(self, imFtr, disFtr, lblList):
        if not self.lossWFlag:
            loss = self.forwardRank(imFtr, disFtr, lblList)
        else:
            loss = self.forwardRankW(imFtr, disFtr, lblList)
        return loss

    def forwardRank(self, imFtr, disFtr, lblList):
        disFtr = disFtr.squeeze()
        bSize = len(lblList)
        if(len(lblList)==1):
            return torch.zeros(1).cuda()
        #pdb.set_trace()
        if self.keepKeyFrameOnly:
            bSize, frmSize, prpSize, ftrSize = imFtr.shape
            #pdb.set_trace()
            keyIdx = (frmSize-1)/2
            imFtr = imFtr[:, keyIdx, :, :]
            imFtr = imFtr.contiguous()
        imFtr = imFtr.view(-1, imFtr.shape[2])
        simMM = torch.mm(imFtr, disFtr.transpose(0, 1))
        simMMRe = simMM.view(bSize, -1, bSize)
        simMax, maxIdx= torch.max(simMMRe, dim=1)
        loss = torch.zeros(1).cuda()
        pairNum = 0
        for i, lblIm in enumerate(lblList):
            posSim = simMax[i, i]
            for j, lblTxt in enumerate(lblList):
                if(lblIm==lblTxt):
                    continue
                else:
                    tmpLoss = simMax[i, j] - posSim + self.margin
                    pairNum +=1
                    if(tmpLoss>0):
                        loss +=tmpLoss
        loss = loss/pairNum

        if self.biLossFlag:
            lossBi = torch.zeros(1).cuda()
            pairNum =0
            for i, lblTxt in enumerate(lblList):
                posSim = simMax[i, i]
                for j, lblIm in enumerate(lblList):
                    if(lblIm==lblTxt):
                        continue
                    else:
                        tmpLoss = simMax[j, i] - posSim + self.margin
                        pairNum +=1
                        if(tmpLoss>0):
                            lossBi +=tmpLoss
            if pairNum>0:
                loss +=lossBi/pairNum
        return loss

    def forwardRankW(self, imFtr, disFtr, lblList):
        disFtr = disFtr.squeeze()
        bSize = len(lblList)
        if(len(lblList)==1):
            return torch.zeros(1).cuda()
#        pdb.set_trace()
        imFtr = imFtr.view(-1, imFtr.shape[2])
        simMM = torch.mm(imFtr, disFtr.transpose(0, 1))
        simMMRe = simMM.view(bSize, -1, bSize)
        simMax, maxIdx= torch.max(simMMRe, dim=1)
        simMax = F.sigmoid(simMax)
        DMtr = -2*torch.log10(simMax)
        loss = torch.zeros(1).cuda()
        pairNum = 0
        for i, lblIm in enumerate(lblList):
            posSim = simMax[i, i]
            for j, lblTxt in enumerate(lblList):
                if(lblIm==lblTxt):
                    continue
                else:
                    tmpLoss = simMax[i, j] - posSim + self.margin
                    pairNum +=1
                    if(tmpLoss>0):
                        loss +=tmpLoss*self.lamda*simMax[i, j]
                    loss +=(1-self.lamda)*DMtr[i, j]
        loss = loss/pairNum

        if self.biLossFlag:
            lossBi = torch.zeros(1).cuda()
            pairNum =0
            for i, lblTxt in enumerate(lblList):
                posSim = simMax[i, i]
                for j, lblIm in enumerate(lblList):
                    if(lblIm==lblTxt):
                        continue
                    else:
                        tmpLoss = simMax[j, i] - posSim + self.margin
                        pairNum +=1
                        if(tmpLoss>0):
                            lossBi +=tmpLoss*self.lamda*simMax[i, j]
            if pairNum>0:
                loss +=lossBi/pairNum
        return loss

class lossGroundR(nn.Module):
    def __init__(self):
        super(lossGroundR, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, logMat, wordLbl):
        loss = 0
        bSize = len(wordLbl)
        for i in range(bSize):
            wL = len(wordLbl[i])
            tmpPredict = logMat[i, :wL, :]
            loss += self.criterion(tmpPredict, torch.LongTensor(wordLbl[i]).cuda())
        return loss

def build_lossEval(opts):
    if opts.wsMode == 'rank':
        return lossEvaluator(opts.margin, opts.biLoss, opts.lossW, opts.lamda)
    elif opts.wsMode == 'groundR':
        return lossGroundR()
    elif opts.wsMode == 'graphSpRank' or opts.wsMode== 'graphSpRankC':
        return lossEvaluator(opts.margin, opts.biLoss, opts.lossW, opts.lamda, keepKeyFrameOnly=True)
