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
    def __init__(self, margin=0.1, biLossFlag=True, lossWFlag=False, lamda=0.8, struct_flag=False, struct_only_flag=False, entropy_regu_flag=False, lamda2=0.1, loss_type=''):
        super(lossEvaluator, self).__init__()
        self.margin =margin
        self.biLossFlag = biLossFlag
        self.lossWFlag = lossWFlag
        self.lamda= lamda
        self.struct_flag = struct_flag
        self.struct_only_flag = struct_only_flag
        self.entropy_regu_flag = entropy_regu_flag
        self.lamda2 = lamda2
        self.loss_type = loss_type 
        if self.entropy_regu_flag:
            self.entropy_calculator = HLoss()

    def forward(self, imFtr=None, disFtr=None, lblList=None, simMM=None, region_gt_ori=None):
        if self.wsMode=='coAtt':
            loss = self.forwardCoAtt_efficient(simMM, lblList)
        return loss

    def forwardCoAtt_efficient(self, simMMRe, lblList):
        
        simMax, maxIdx= torch.max(simMMRe.squeeze(3), dim=1)
        loss = torch.zeros(1).cuda()
        pair_num = 0.000001

        t1 = time.time()
        #print(lblList)
        bSize = len(lblList)
        b_size = simMax.shape[0]
        pos_diag = torch.cat([simMax[i, i].unsqueeze(0) for i in range(b_size)])
        one_mat = torch.ones(b_size, b_size).cuda()
        pos_diag_mat = torch.mul(pos_diag, one_mat)
        pos_diag_trs = pos_diag_mat.transpose(0,1)

        mask_val = torch.ones(b_size, b_size).cuda()

        for i in range(b_size):
            lblI = lblList[i]
            for j in range(i, b_size):
                lblJ = lblList[j]
                if lblI==lblJ:
                    mask_val[i, j]=0
                    mask_val[j, i]=0
        pair_num = pair_num + torch.sum(mask_val)
        
        loss_mat_1 = simMax -pos_diag + self.margin 
        loss_mask = (loss_mat_1>0).float()
        loss_mat_1_mask = loss_mat_1 *loss_mask * mask_val 
        loss1 = torch.sum(loss_mat_1_mask)
        
        loss_mat_2 = simMax -pos_diag_trs +self.margin
        loss_mask_2 = (loss_mat_2>0).float()
        loss_mat_2_mask = loss_mat_2 *loss_mask_2 * mask_val 
        loss2 = torch.sum(loss_mat_2_mask)
        loss =  (loss1+loss2)/pair_num

        if self.entropy_regu_flag:
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

def build_lossEval(opts):
    if opts.wsMode == 'rankTube' or opts.wsMode=='coAtt' or opts.wsMode=='coAttV2' or opts.wsMode=='coAttV3' or opts.wsMode == 'coAttV4' or  opts.wsMode=='rankFrm' or opts.wsMode=='coAttBi' or opts.wsMode=='coAttBiV2' or opts.wsMode =='coAttBiV3':
        loss_criterion = lossEvaluator(opts.margin, opts.biLoss, opts.lossW, \
                opts.lamda, opts.struct_flag, opts.struct_only, \
                opts.entropy_regu_flag, opts.lamda2, \
                loss_type=opts.loss_type)
        loss_criterion.wsMode =opts.wsMode
        return loss_criterion
    elif opts.wsMode =='rankGroundR' or opts.wsMode =='coAttGroundR' or opts.wsMode=='rankGroundRV2':
        loss_criterion = lossGroundR(entropy_regu_flag=opts.entropy_regu_flag, \
               lamda2=opts.lamda2 )
        loss_criterion.wsMode = opts.wsMode
        return loss_criterion
