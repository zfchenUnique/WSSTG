import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from multiGraphAttention import *


class wsEmb(nn.Module):
    def __init__(self, imEncoder, wordEncoder):
        super(wsEmb, self).__init__()
        self._train=True
        self.imEncoder = imEncoder
        self.wordEncoder = wordEncoder
        self._initialize_weights()
        self.wsMode = 'rank'

    def forward(self, imDis, wordEmb, capLengthsFull=None, frmListFull=None, rpListFull=None, dataIdx=None):
        if dataIdx is not None:
            numEle = len(dataIdx)
            if capLengthsFull is not None:
                capLengths = [capLengthsFull[int(dataIdx[i])] for i in range(numEle)]
            else:
                capLengths = capLengthsFull
            if frmListFull is not None:
                frmList = [frmListFull[int(dataIdx[i])] for i in range(numEle)]
            else:
                frmList = frmListFull
            if rpListFull is not None:
                rpList = [rpListFull[int(dataIdx[i])] for i in range(numEle)]
            else:
                rpList = rpListFull
        else:
            capLengths = capLengthsFull
            frmList = frmListFull
            rpList = rpListFull

        #pdb.set_trace()
        if  self.wsMode == 'rank':
            imEnDis, wordEnDis = self.forwardRank(imDis, wordEmb, capLengths)
            return imEnDis, wordEnDis
        elif self.wsMode == 'groundR':
            logDist, rpSS= self.forwardGroundR(imDis, wordEmb, capLengths, frmList)
            return logDist, rpSS
        elif self.wsMode == 'graphSpRank' or self.wsMode == 'graphSpRankC':
            return self.forwardGraphSpRank(imDis, wordEmb, capLengths, rpList)

    def forwardRank(self, imDis, wordEmb, capLengths):
        assert len(imDis.size())==3
        assert len(wordEmb.size())==3
        imEnDis = self.imEncoder(imDis)
        wordEnDis = self.wordEncoder(wordEmb, capLengths)
        assert len(imEnDis.size())==3
        assert len(wordEnDis.size())==2
        imEnDis = F.normalize(imEnDis, p=2, dim=1)
        wordEnDis = F.normalize(wordEnDis, p=2, dim=3)
        return imEnDis, wordEnDis

    def forwardGraphSpRank(self, imDis, wordEmb, capLengths, rpList):
        assert len(imDis.size())==4
        assert len(wordEmb.size())==3
        wordEnDis = self.wordEncoder(wordEmb, capLengths)
        wordEnDis = F.normalize(wordEnDis, p=2, dim=1)
        imEnDis, aff_softmax, aff_scale, aff_weight = self.imEncoder(imDis, rpList, wordEnDis)
        #imEnDis = self.imEncoder(imDis, rpList, wordEnDis)
        assert len(imEnDis.size())==4
        assert len(imEnDis.size())==4
        assert len(wordEnDis.size())==2
        imEnDis = F.normalize(imEnDis, p=2, dim=3)
        # keep only keyframe
        return imEnDis, wordEnDis, aff_softmax, aff_scale, aff_weight

    def forwardGroundR(self, imDis, wordEmb, capLengths, frmList=None):
        assert len(imDis.size())==3
        assert len(wordEmb.size())==3
        imEnDis = self.imEncoder(imDis)
        wordEnDis = self.wordEncoder(wordEmb, capLengths)
        assert len(imDis.size())==3
        assert len(wordEmb.size())==3
        #imEnDis = F.normalize(imEnDis, p=2, dim=2)
        #wordEnDis = F.normalize(wordEnDis, p=2, dim=2)
        visFtrEm, rpSS= self.visDecoder(imEnDis, wordEnDis, frmList)
        reCnsSen =None
        if self.training:
            reCnsSen= self.recntr(visFtrEm, wordEmb, capLengths)
        return reCnsSen, rpSS

    def _initialize_weights(self):
        #pdb.set_trace()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_groundR(self, opts):
        self.wsMode = 'groundR'
        self.recntr = build_recontructor(opts)
        self.visDecoder = build_visDecoder(opts)    

def build_vis_encoder(opts):
    inputDim = 2048
    visNet = torch.nn.Sequential(
            torch.nn.Linear(inputDim, inputDim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(inputDim, opts.dim_ftr),
            )
    return visNet



class txtEncoder(nn.Module):
    def __init__(self, embedDim, hidden_dim):
        super(txtEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm =nn.LSTM(embedDim, hidden_dim, batch_first=True)
        self.hidden = self.init_hidden()
        
    def init_hidden(self, batchSize=10):
        self.hidden=(torch.zeros(1, batchSize, self.hidden_dim).cuda(),
                torch.zeros(1, batchSize, self.hidden_dim).cuda())

    def forward(self, wordMatrix, wordLeg=None):
#        pdb.set_trace()
        # shorten steps for faster training
        self.init_hidden(wordMatrix.shape[0])
        lstmOut, self.hidden = self.lstm(wordMatrix, self.hidden)
        #pdb.set_trace()
        # lstmOut: B*T*D
        if wordLeg==None:
            return self.hidden[0]
        else:
            txtEMb = [lstmOut[i, wordLeg[i]-1,:] for i in range(len(wordLeg))]
            #pdb.set_trace()
            txtEmbMat = torch.stack(txtEMb)
            return txtEmbMat

class txtDecoder(nn.Module):
    def __init__(self, embedDim, hidden_dim, vocaSize):
        super(txtDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm =nn.LSTM(embedDim, hidden_dim, batch_first=True)
        self.hidden = self.init_hidden()
        self.dec_log = torch.nn.Linear(hidden_dim, vocaSize)
        
    def init_hidden(self, batchSize=10):
        self.hidden=(torch.zeros(1, batchSize, self.hidden_dim).cuda(),
                torch.zeros(1, batchSize, self.hidden_dim).cuda())

    def forward(self, visFtr, capFtr, capLght):
        inputs = torch.cat((visFtr.unsqueeze(1), capFtr), 1)

        # pack data (prepare it for pytorch model)
       # inputs_packed = pack_padded_sequence(inputs, capLght, batch_first=True)
        inputs_packed = inputs
        # run data through recurrent network
        hiddens, _ = self.lstm(inputs_packed)
        #pdb.set_trace()
        outputs = self.dec_log(hiddens)
        #pdb.set_trace()
        return outputs


class visDecoder(nn.Module):
    def __init__(self, dim_ftr, hdSize):
        super(visDecoder, self).__init__()
        self.attNet = torch.nn.Sequential(
            torch.nn.Linear(dim_ftr*2, hdSize),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hdSize, 1),
            )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, visRPFtr, capFtr, frmList=None):
        layers, bSize, dim = capFtr.shape
        capFtr = capFtr.permute(1, 0, 2)
        capFtr = capFtr.view(bSize, 1, -1)
        bSizeV, rpNum, dimV =visRPFtr.shape
        capFtrExp = capFtr.expand(bSize, rpNum, dim)
        # when testing, handling multi-frames for one caption
        #pdb.set_trace()
        if bSizeV>bSize:
            visFtrAtt = None
            stId =0
            rpScoreList = list()
            for i in range(bSize):
                numFtr = len(frmList[i])
                subCapFtrExp = capFtrExp[i, :, :].unsqueeze(0).expand(numFtr, rpNum, dim)
                endId = stId + numFtr
                subVisRPFtr = visRPFtr[stId:endId, :, :]
                subConcaFtr = torch.cat((subCapFtrExp, subVisRPFtr), 2)
                rpScore = self.attNet(subConcaFtr)
                rpScore = rpScore.view(numFtr, rpNum)
                rpSS= F.softmax(rpScore, dim=1) 
                stId = endId
                rpScoreList.append(rpSS)
            rpScoreMat= torch.cat(rpScoreList, 0) 
            return visFtrAtt, rpScoreMat

        concaFtr = torch.cat((capFtrExp, visRPFtr), 2)
        rpScore = self.attNet(concaFtr)
        rpScore = rpScore.view(bSize, rpNum)
        assert(len(rpScore.shape)==2)
        rpSS= F.softmax(rpScore, dim=1) 
        rpSS = rpSS.viewbSize, rpNum, 1)
        visFtrAtt = torch.sum(torch.mul(visRPFtr, rpSS), dim=1)
        return visFtrAtt, rpSS 

def build_visDecoder(opts):
    return visDecoder(opts.dim_ftr, opts.hdSize)

def build_txt_encoder(opts):
    embedDim =300
    txt_encoder =  txtEncoder(embedDim, opts.dim_ftr)
    return txt_encoder 

def build_recontructor(opts):
    return txtDecoder(opts.dim_ftr, opts.hdSize, opts.vocaSize)

class visEncoderAtt(nn.Module):
    def __init__(self, dim_input, hdSize, ftrSize, mdNum, attType='sp'):
        super(visEncoderAtt, self).__init__()
        self.attType = attType
        self.mdList = nn.ModuleList()
        if self.attType == 'sp':
            for i in range(mdNum):
                self.mdList.append(nn.Linear(dim_input, hdSize))
                dim_input = hdSize
                self.mdList.append(multiHeadAttention( ftrDimList=[hdSize, 64, hdSize], \
                        groupNum=16, kNN=None, atType=attType))
            self.mdList.append(nn.Linear(hdSize, ftrSize))
        elif self.attType == 'spc':
            for i in range(mdNum):
                self.mdList.append(nn.Linear(dim_input, hdSize))
                dim_input = hdSize*2 # for conca
                self.mdList.append(multiHeadAttention( ftrDimList=[hdSize, 64, hdSize], \
                        groupNum=16, kNN=None, atType=attType))
            self.mdList.append(nn.Linear(hdSize*2, ftrSize))


    def forward(self, visFtr, rpList, capFtr=None):
        if self.attType == 'sp':
            return self.forwardSp(visFtr, rpList)
        if self.attType == 'spc':
            return self.forwardSpC(visFtr, rpList)
        

    def forwardSp(self, visFtr, rpList):
        bSize, frmSize, prpSize, ftrSize = visFtr.shape
        visFtr = visFtr.view(-1, ftrSize)
        for i, subMd  in enumerate(self.mdList):
            if isinstance(subMd, multiHeadAttention):
                visFtr = visFtr.view(bSize, frmSize, prpSize, -1)
                #x_relation = subMd(visFtr, rpPropM=rpList)
                x_relation, aff_softmax, aff_scale, aff_weight = subMd(visFtr, rpPropM=rpList)
                visFtr  =  visFtr +x_relation.view_as(visFtr)
            else:
                visFtr = visFtr.view(bSize*frmSize*prpSize, -1)
                visFtr = subMd(visFtr)
                visFtr = F.relu(visFtr)
                if subMd.training:
                    visFtr = F.dropout(visFtr)
        visFtr = visFtr.view(bSize, frmSize, prpSize, -1)       
        return visFtr, aff_softmax, aff_scale, aff_weight

    def forwardSpC(self, visFtr, rpList):
        bSize, frmSize, prpSize, ftrSize = visFtr.shape
        visFtr = visFtr.view(-1, ftrSize)
        for i, subMd  in enumerate(self.mdList):
            if isinstance(subMd, multiHeadAttention):
                visFtr = visFtr.view(bSize, frmSize, prpSize, -1)
                #x_relation = subMd(visFtr, rpPropM=rpList)
                x_relation, aff_softmax, aff_scale, aff_weight = subMd(visFtr, rpPropM=rpList)
                visFtr  =  torch.cat((visFtr, x_relation.view_as(visFtr)), 3)
            else:
                visFtr = visFtr.view(bSize*frmSize*prpSize, -1)
                visFtr = subMd(visFtr)
                visFtr = F.relu(visFtr)
                if subMd.training:
                    visFtr = F.dropout(visFtr)
        visFtr = visFtr.view(bSize, frmSize, prpSize, -1)       
        return visFtr, aff_softmax, aff_scale, aff_weight



def build_vis_encoderAtt(opts):
    if opts.wsMode=='graphSpRank':
        attTpye = 'sp'
    elif opts.wsMode=='graphSpRankC':
        attTpye = 'spc'
    return visEncoderAtt(2048, opts.moduleHdSize, opts.dim_ftr, opts.moduleNum, attTpye)

def build_network(opts):
    if opts.wsMode =='groundR':
        imEncoder= build_vis_encoder(opts)
        wordEncoder = build_txt_encoder(opts)
        wsEncoder = wsEmb(imEncoder, wordEncoder)
        wsEncoder.build_groundR(opts)
    elif opts.wsMode =='rank': 
        imEncoder= build_vis_encoder(opts)
        wordEncoder = build_txt_encoder(opts)
        wsEncoder = wsEmb(imEncoder, wordEncoder)
    elif opts.wsMode =='graphSpRank' or opts.wsMode == 'graphSpRankC':
        imEncoder= build_vis_encoderAtt(opts)
        wordEncoder = build_txt_encoder(opts)
        wsEncoder = wsEmb(imEncoder, wordEncoder)
        wsEncoder.wsMode = opts.wsMode
    
    if opts.gpu:
        wsEncoder= wsEncoder.cuda()
    if opts.initmodel is not None:
        md_stat = torch.load(opts.initmodel)
        wsEncoder.load_state_dict(md_stat)
    if opts.isParal:
        wsEncoder = nn.DataParallel(wsEncoder).cuda()
    
    return wsEncoder












