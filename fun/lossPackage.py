import torch 
import torch.nn as nn
import torch.nn.functional as F
import pdb

class lossEvaluator(nn.Module):
    def __init__(self, margin=0.1, biLossFlag=True, lossWFlag=False, lamda=0.8, struct_flag=False):
        super(lossEvaluator, self).__init__()
        self.margin =margin
        self.biLossFlag = biLossFlag
        self.lossWFlag = lossWFlag
        self.lamda= lamda
        self.struct_flag = struct_flag

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
        imFtr = imFtr.view(-1, imFtr.shape[2])
        simMM = torch.mm(imFtr, disFtr.transpose(0, 1))
        simMMRe = simMM.view(bSize, -1, bSize)
        simMax, maxIdx= torch.max(simMMRe, dim=1)
        loss = torch.zeros(1).cuda()
        pairNum = 0.000001
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
            pairNum =0.000001
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

        if self.struct_flag:
           # select most similar regions from the tubes
            imFtr = imFtr.view(bSize, -1, imFtr.shape[1])
            tube_ftr_list = [ imFtr[i, int(maxIdx[i, i]), ...] for i in range(bSize)]
            tube_ftr_tensor = torch.stack(tube_ftr_list, 0)
            vis_ftr_struct_sim = torch.mm(tube_ftr_tensor, tube_ftr_tensor.transpose(0, 1))
            txt_ftr_struct_sim = torch.mm(disFtr, disFtr.transpose(0, 1))
            sim_res_mat = vis_ftr_struct_sim - txt_ftr_struct_sim
            sim_res_mat_pow2 = torch.pow(sim_res_mat, 2)
            sim_res_mat_pow2_sqrt = torch.sqrt(sim_res_mat_pow2 + torch.ones(bSize, bSize).cuda()*0.000001)
            #sim_res_mat_pow2_sqrt = sim_res_mat_pow2
            sim_res_l2_loss = sim_res_mat_pow2_sqrt.sum()/(bSize*bSize-bSize)
            print('biloss %3f, struct loss %3f\n' %(float(loss), float(sim_res_l2_loss)))            
            loss += self.lamda*sim_res_l2_loss
            #pdb.set_trace()

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

def build_lossEval(opts):
    if opts.wsMode == 'rankTube':
        return lossEvaluator(opts.margin, opts.biLoss, opts.lossW, opts.lamda, opts.struct_flag)
