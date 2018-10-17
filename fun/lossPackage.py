import torch 
import torch.nn as nn
import torch.nn.functional as F
import pdb
import time

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        b = torch.mean(b)
        return b

class lossEvaluator(nn.Module):
    def __init__(self, margin=0.1, biLossFlag=True, lossWFlag=False, lamda=0.8, struct_flag=False, struct_only_flag=False, entropy_regu_flag=False, lamda2=0.1):
        super(lossEvaluator, self).__init__()
        self.margin =margin
        self.biLossFlag = biLossFlag
        self.lossWFlag = lossWFlag
        self.lamda= lamda
        self.struct_flag = struct_flag
        self.struct_only_flag = struct_only_flag
        self.entropy_regu_flag = entropy_regu_flag
        self.lamda2 = lamda2
        if self.entropy_regu_flag:
            self.entropy_calculator = HLoss()

    def forward(self, imFtr=None, disFtr=None, lblList=None, simMM=None):
        if self.wsMode=='rankTube':
            loss = self.forwardRank(imFtr, disFtr, lblList)
        elif self.wsMode=='coAtt':
            loss = self.forwardCoAtt(simMM, lblList)
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

#        pdb.set_trace()
        t1 = time.time()
        #print(lblList)
        if not self.struct_only_flag:
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

        if self.biLossFlag and not self.struct_only_flag:
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
        t2 = time.time()
        #print(t2-t1)
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
            print('biloss %3f, struct loss %3f ' %(float(loss), float(sim_res_l2_loss)))            
            loss += self.lamda*sim_res_l2_loss

        if self.entropy_regu_flag:
            #simMMRe = simMM.view(bSize, -1, bSize)
            # simMMRe: bSize, prpNum, bSize
            ftr_match_pair_list = list()
            ftr_unmatch_pair_list = list()

            for i in range(bSize):
                for j in range(bSize):
                    if i==j:
                        ftr_match_pair_list.append(simMMRe[i, ..., i])
                    elif lblList[i]!=lblList[j]:
                        ftr_unmatch_pair_list.append(simMMRe[i, ..., j])

            ftr_match_pair_mat = torch.stack(ftr_match_pair_list, 0)
            ftr_unmatch_pair_mat = torch.stack(ftr_unmatch_pair_list, 0)
            match_num = len(ftr_match_pair_list)
            unmatch_num = len(ftr_unmatch_pair_list)
            if match_num>0:
                entro_loss = self.entropy_calculator(ftr_match_pair_mat)
            loss +=self.lamda2*entro_loss 
            print('entropy loss: %3f ' %(float(entro_loss)))
            #pdb.set_trace()
        print('\n')
        return loss

    def forwardCoAtt(self, simMMRe, lblList):
        bSize = len(lblList)
        if(len(lblList)==1):
            return torch.zeros(1).cuda()
        simMax, maxIdx= torch.max(simMMRe, dim=1)
        loss = torch.zeros(1).cuda()
        pairNum = 0.000001

#        pdb.set_trace()
        t1 = time.time()
        #print(lblList)
        if not self.struct_only_flag:
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

        if self.biLossFlag and not self.struct_only_flag:
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
        t2 = time.time()

        if self.entropy_regu_flag:
            #simMMRe = simMM.view(bSize, -1, bSize)
            # simMMRe: bSize, prpNum, bSize
            simMMRe = simMMRe.squeeze()
            ftr_match_pair_list = list()
            ftr_unmatch_pair_list = list()

            for i in range(bSize):
                for j in range(bSize):
                    if i==j:
                        ftr_match_pair_list.append(simMMRe[i, ..., i])
                    elif lblList[i]!=lblList[j]:
                        ftr_unmatch_pair_list.append(simMMRe[i, ..., j])

            ftr_match_pair_mat = torch.stack(ftr_match_pair_list, 0)
            ftr_unmatch_pair_mat = torch.stack(ftr_unmatch_pair_list, 0)
            match_num = len(ftr_match_pair_list)
            unmatch_num = len(ftr_unmatch_pair_list)
            if match_num>0:
                entro_loss = self.entropy_calculator(ftr_match_pair_mat)
            loss +=self.lamda2*entro_loss 
            print('entropy loss: %3f ' %(float(entro_loss)))
            #pdb.set_trace()
        print('\n')
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
    if opts.wsMode == 'rankTube' or opts.wsMode=='coAtt':
        loss_criterion = lossEvaluator(opts.margin, opts.biLoss, opts.lossW, \
                opts.lamda, opts.struct_flag, opts.struct_only, \
                opts.entropy_regu_flag, opts.lamda2)
        loss_criterion.wsMode =opts.wsMode
        return loss_criterion
