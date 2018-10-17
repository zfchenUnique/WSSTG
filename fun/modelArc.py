import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from multiGraphAttention import *
from netvlad import NetVLAD
from classSST import *

class wsEmb(nn.Module):
    def __init__(self, imEncoder, wordEncoder):
        super(wsEmb, self).__init__()
        self._train=True
        self.imEncoder = imEncoder
        self.wordEncoder = wordEncoder
        self._initialize_weights()
        self.wsMode = 'rank'
        self.vis_type = 'fc'

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
        
        if  self.wsMode == 'coAtt':
            simMM = self.forwardCoAtt(imDis, wordEmb, capLengths)
            return simMM

    def forwardRank(self, imDis, wordEmb, capLengths):
        if self.vis_type =='fc':
            imDis = imDis.mean(1)
        else:
            assert len(imDis.size())==3
        assert len(wordEmb.size())==3
        #pdb.set_trace()
        imEnDis = self.imEncoder(imDis)
        wordEnDis = self.wordEncoder(wordEmb, capLengths)
        assert len(imEnDis.size())==2
        assert len(wordEnDis.size())==2
        imEnDis = F.normalize(imEnDis, p=2, dim=1)
        wordEnDis = F.normalize(wordEnDis, p=2, dim=1)
        return imEnDis, wordEnDis

    def forwardCoAtt(self, imDis, wordEmb, capLengths):
        assert len(imDis.size())==3
        assert len(wordEmb.size())==3
        #pdb.set_trace()
        simMM = self.imEncoder(imDis, wordEmb, capLengths)
        return simMM

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

class vlad_encoder(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, centre_num, alpha=1.0):
        super(vlad_encoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.net_vlad = NetVLAD(num_clusters= centre_num, dim= hidden_dim, alpha=1.0)
        self.fc2 = torch.nn.Linear(centre_num*hidden_dim, out_dim)

    def forward(self, input_data):
        input_hidden = self.fc1(input_data)
        input_hidden = F.relu(input_hidden)
        input_hidden = input_hidden.unsqueeze(dim=3)
        input_hidden = input_hidden.transpose(dim0=1, dim1=2)
        #pdb.set_trace()
        input_vlad = self.net_vlad(input_hidden)
        #pdb.set_trace()
        out_vlad = self.fc2(input_vlad)
        out_vlad = F.relu(out_vlad)
        return out_vlad

def build_vis_vlad_encoder_v1(opts):
    input_dim = opts.vis_dim
    hidden_dim = opts.hidden_dim
    centre_num = opts.centre_num
    out_dim = opts.dim_ftr
    alpha = opts.vlad_alpha
    vis_encoder  = vlad_encoder(input_dim, out_dim, hidden_dim, centre_num, alpha=1.0 ) 
    return vis_encoder

def build_vis_seq_encoder(opts):
    embedDim = opts.vis_dim
    if opts.vis_type == 'lstm' or opts.vis_type=='gru':
        vis_seq_encoder =  visSeqEncoder(embedDim, opts.dim_ftr, opts.vis_type)
        return vis_seq_encoder
    elif opts.vis_type == 'fc':
        vis_avg_encoder = build_vis_fc_encoder(opts)
        return vis_avg_encoder
    elif opts.vis_type == 'vlad_v1':
        vis_vlad_encoder = build_vis_vlad_encoder_v1(opts)
        return vis_vlad_encoder

def build_network(opts):
    if opts.wsMode == 'rankTube': 
        imEncoder= build_vis_seq_encoder(opts)
        wordEncoder = build_txt_encoder(opts)
        wsEncoder = wsEmb(imEncoder, wordEncoder)
    elif opts.wsMode == 'coAtt':
        sst_Obj = SST(opts)
        wsEncoder = wsEmb(sst_Obj, None)
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

