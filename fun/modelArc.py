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

        if  self.wsMode == 'rankTube':
            imEnDis, wordEnDis = self.forwardRank(imDis, wordEmb, capLengths)
            return imEnDis, wordEnDis

    def forwardRank(self, imDis, wordEmb, capLengths):
        #pdb.set_trace()
        if self.vis_type =='fc':
            imDis = imDis.mean(1)
        else:
            assert len(imDis.size())==3
        assert len(wordEmb.size())==3
        imEnDis = self.imEncoder(imDis)
        wordEnDis = self.wordEncoder(wordEmb, capLengths)
        assert len(imEnDis.size())==2
        assert len(wordEnDis.size())==2
        imEnDis = F.normalize(imEnDis, p=2, dim=1)
        wordEnDis = F.normalize(wordEnDis, p=2, dim=1)
        return imEnDis, wordEnDis

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

class txtEncoder(nn.Module):
    def __init__(self, embedDim, hidden_dim, seq_type='lstm'):
        super(txtEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_type = seq_type
        if seq_type =='lstm':
            self.lstm =nn.LSTM(embedDim, hidden_dim, batch_first=True)
        elif seq_type =='gru':
            self.lstm =nn.GRU(embedDim, hidden_dim, batch_first=True)
        self.hidden = self.init_hidden()
        
    def init_hidden(self, batchSize=10):
        if self.seq_type=='lstm':
            self.hidden=(torch.zeros(1, batchSize, self.hidden_dim).cuda(),
                    torch.zeros(1, batchSize, self.hidden_dim).cuda())
        elif self.seq_type=='gru':
            self.hidden=torch.zeros(1, batchSize, self.hidden_dim).cuda()


    def forward(self, wordMatrix, wordLeg=None):
        #pdb.set_trace()
        # shorten steps for faster training
        self.init_hidden(wordMatrix.shape[0])
        #pdb.set_trace()
        lstmOut, self.hidden = self.lstm(wordMatrix, self.hidden)
        #pdb.set_trace()
        # lstmOut: B*T*D
        if wordLeg==None:
            return self.hidden[0].squeeze()
        else:
            txtEMb = [lstmOut[i, wordLeg[i]-1,:] for i in range(len(wordLeg))]
            #pdb.set_trace()
            txtEmbMat = torch.stack(txtEMb)
            return txtEmbMat

def build_txt_encoder(opts):
    embedDim = 300
    txt_encoder =  txtEncoder(embedDim, opts.dim_ftr, opts.txt_type)
    return txt_encoder 

class visSeqEncoder(nn.Module):
    def __init__(self, embedDim, hidden_dim, seq_Type='lstm'):
        super(visSeqEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_type = seq_Type
        if seq_Type =='lstm':
            self.lstm =nn.LSTM(embedDim, hidden_dim, batch_first=True)
        elif seq_Type =='gru':
            self.lstm =nn.GRU(embedDim, hidden_dim, batch_first=True)

        self.hidden = self.init_hidden()
        
    def init_hidden(self, batchSize=10):
        if self.seq_type=='lstm':
            self.hidden=(torch.zeros(1, batchSize, self.hidden_dim).cuda(),
                    torch.zeros(1, batchSize, self.hidden_dim).cuda())
        elif self.seq_type=='gru':
            self.hidden=torch.zeros(1, batchSize, self.hidden_dim).cuda()

    def forward(self, wordMatrix, wordLeg=None):
        # shorten steps for faster training
        self.init_hidden(wordMatrix.shape[0])
        #pdb.set_trace()
        lstmOut, self.hidden = self.lstm(wordMatrix, self.hidden)
        #pdb.set_trace()
        # lstmOut: B*T*D
        if wordLeg==None:
            return self.hidden[0].squeeze()
        else:
            txtEMb = [lstmOut[i, wordLeg[i]-1,:] for i in range(len(wordLeg))]
            #pdb.set_trace()
            txtEmbMat = torch.stack(txtEMb)
            return txtEmbMat

def build_vis_fc_encoder(opts):
    inputDim = opts.vis_dim 
    visNet = torch.nn.Sequential(
            torch.nn.Linear(inputDim, inputDim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(inputDim, opts.dim_ftr),
            )
    return visNet


def build_vis_seq_encoder(opts):
    embedDim = opts.vis_dim
    if opts.vis_type == 'lstm' or opts.vis_type=='gru':
        vis_seq_encoder =  visSeqEncoder(embedDim, opts.dim_ftr, opts.vis_type)
        return vis_seq_encoder
    elif opts.vis_type == 'fc':
        vis_avg_encoder = build_vis_fc_encoder(opts)
        return vis_avg_encoder


def build_network(opts):
    if opts.wsMode =='rankTube': 
        imEncoder= build_vis_seq_encoder(opts)
        wordEncoder = build_txt_encoder(opts)
        wsEncoder = wsEmb(imEncoder, wordEncoder)
    wsEncoder.wsMode = opts.wsMode
    wsEncoder.vis_type = opts.vis_type
    if opts.gpu:
        wsEncoder= wsEncoder.cuda()
    if opts.initmodel is not None:
        md_stat = torch.load(opts.initmodel)
        wsEncoder.load_state_dict(md_stat)
    if opts.isParal:
        wsEncoder = nn.DataParallel(wsEncoder).cuda()
    
    return wsEncoder

