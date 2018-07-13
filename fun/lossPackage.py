import torch 
import torch.nn as nn
import pdb

class lossEvaluator(nn.Module):
    def __init__(self, margin=0.1):
        super(lossEvaluator, self).__init__()
        self.margin =margin
    
    def forward(self, imFtr, disFtr, lblList):
        disFtr = disFtr.squeeze()
        bSize = len(lblList)
        if(len(lblList)==1):
            return torch.zeros(1).cuda()
        #pdb.set_trace()
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

        return loss/pairNum


def build_lossEval(opts):
    return lossEvaluator(opts.margin)

