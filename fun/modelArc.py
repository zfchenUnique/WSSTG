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

        if  self.wsMode == 'rankTube' or self.wsMode =='rankFrm':
            imEnDis, wordEnDis = self.forwardRank(imDis, wordEmb, capLengths)
            return imEnDis, wordEnDis
       
        #pdb.set_trace()
        if  self.wsMode == 'coAtt' or self.wsMode == 'coAttV2' or self.wsMode=='coAttV3' or self.wsMode=='coAttV4' or self.wsMode=='coAttBi':
            simMM = self.forwardCoAtt(imDis, wordEmb, capLengths)
            return simMM

        elif self.wsMode == 'rankGroundR' or self.wsMode=='rankGroundRV2':
            logDist, rpSS= self.forwardGroundR(imDis, wordEmb, capLengths)
            return logDist, rpSS
        
        elif self.wsMode == 'coAttGroundR':
            logDist, rpSS= self.forwardCoAttGroundR(imDis, wordEmb, capLengths)
            return logDist, rpSS

    
    def forwardCoAttGroundR(self, imDis, wordEmb, capLengths):
        b_size = imDis.shape[0]
        assert len(imDis.size())==4
        assert len(wordEmb.size())==4

        imDis = imDis.view(-1, imDis.shape[2], imDis.shape[3])
        wordEmb = wordEmb.view(-1, wordEmb.shape[2], wordEmb.shape[3])
        imEnDis, wordEnDis = self.imEncoder(imDis, wordEmb, capLengths)
        
        imEnDis = imEnDis.view(b_size , -1, imEnDis.shape[2])
        wordEnDis = wordEnDis.view(b_size, -1, wordEnDis.shape[2])
        visFtrEm, rpSS= self.visDecoder(imEnDis, wordEnDis)
        reCnsSen =None
        if self.training:
            assert visFtrEm.shape[2]==1
            visFtrEm = visFtrEm.squeeze(dim=2)
            reCnsSen= self.recntr(visFtrEm, wordEmb, capLengths)
        return reCnsSen, rpSS
    
    
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

    def forwardGroundR(self, imDis, wordEmb, capLengths, frmList=None):
        #pdb.set_trace()
        b_size = imDis.shape[0]
        assert len(imDis.size())==4
        assert len(wordEmb.size())==4

        if self.vis_type =='fc':
            imDis = imDis.mean(2)
        else:
            imDis = imDis.view(-1, imDis.shape[2], imDis.shape[3])
        imEnDis = self.imEncoder(imDis)
        wordEmb = wordEmb.view(-1, wordEmb.shape[2], wordEmb.shape[3])
        wordEnDis = self.wordEncoder(wordEmb, capLengths)
        #imEnDis = F.normalize(imEnDis, p=2, dim=2)
        #wordEnDis = F.normalize(wordEnDis, p=2, dim=2)
#        pdb.set_trace()
        if (len(imEnDis.shape)==3):
            imEnDis = imEnDis.view(b_size , -1, imEnDis.shape[2])
        elif(len(imEnDis.shape)==2):
            imEnDis = imEnDis.view(b_size , -1, imEnDis.shape[1])
        wordEnDis = wordEnDis.view(b_size, -1, wordEnDis.shape[1])
        visFtrEm, rpSS= self.visDecoder(imEnDis, wordEnDis)
        
        #pdb.set_trace()
        if hasattr(self, 'fc2recontr'):
            visFtrEm_trs = visFtrEm.transpose(1,2)
            visFtrEm_trs = self.fc2recontr(visFtrEm_trs)
            visFtrEm = visFtrEm_trs.transpose(1,2)
        reCnsSen =None
#        pdb.set_trace()
        if self.training:
            assert visFtrEm.shape[2]==1
            visFtrEm = visFtrEm.squeeze(dim=2)
            reCnsSen= self.recntr(visFtrEm, wordEmb, capLengths)
        return reCnsSen, rpSS


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

    def build_groundR(self, opts):
        self.recntr = build_recontructor(opts)
        self.visDecoder = build_visDecoder(opts)    
        if opts.wsMode=='rankGroundRV2':
            self.fc2recontr = torch.nn.Linear(opts.dim_ftr, 300)

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
        #pdb.set_trace()
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
    def __init__(self, dim_ftr, hdSize, coAtt_flag=False):
        super(visDecoder, self).__init__()
        self.attNet = torch.nn.Sequential(
            torch.nn.Linear(dim_ftr*2, hdSize),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(hdSize, 1),
            )
        self.coAtt_flag = coAtt_flag
    
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

    def forward(self, vis_rp_ftr, cap_ftr):
        b_size, per_cap_num, dim_cap = cap_ftr.shape
        b_szie_v, rp_num, dim_vis =vis_rp_ftr.shape
   
        # for co-Attention
        if self.coAtt_flag:
            conca_ftr = torch.cat((cap_ftr, vis_rp_ftr), 2)
            rpScore = self.attNet(conca_ftr)
            rpScore = rpScore.view(b_size, rp_num)
            rpSS= F.softmax(rpScore, dim=1)
            rpSS = rpSS.view(b_size, rp_num, 1)
            visFtrAtt = torch.sum(torch.mul(vis_rp_ftr, rpSS), dim=1)
            return visFtrAtt.unsqueeze(2), rpSS

        # for independent embedding
        rp_ss_list  = list()
        vis_att_list  = list()
        for i in range(per_cap_num):
            cap_ftr_exp = cap_ftr[:, i, :].unsqueeze(dim=1).expand(b_size, rp_num, dim_cap)
            conca_ftr = torch.cat((cap_ftr_exp, vis_rp_ftr), 2)
            rpScore = self.attNet(conca_ftr)
            rpScore = rpScore.view(b_size, rp_num)
            assert(len(rpScore.shape)==2)
            rpSS= F.softmax(rpScore, dim=1)
            rpSS = rpSS.view(b_size, rp_num, 1)
            visFtrAtt = torch.sum(torch.mul(vis_rp_ftr, rpSS), dim=1)
            rp_ss_list.append(rpSS)
            vis_att_list.append(visFtrAtt.unsqueeze(dim=2))
        vis_ftr_mat = torch.cat(vis_att_list, dim=2)
        rp_ss_mat = torch.cat(rp_ss_list, dim=2)
        return vis_ftr_mat, rp_ss_mat


class txtEncoderV2(nn.Module):
    def __init__(self, embedDim, hidden_dim, seq_type='lstmV2'):
        super(txtEncoderV2, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_type = seq_type
        self.fc1 = torch.nn.Linear(embedDim, hidden_dim)
        if seq_type =='lstmV2':
            self.lstm =nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        elif seq_type =='gruV2':
            self.lstm =nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.hidden = self.init_hidden()
        
    def init_hidden(self, batchSize=10):
        if self.seq_type=='lstmV2':
            self.hidden=(torch.zeros(1, batchSize, self.hidden_dim).cuda(),
                    torch.zeros(1, batchSize, self.hidden_dim).cuda())
        elif self.seq_type=='gruV2':
            self.hidden=torch.zeros(1, batchSize, self.hidden_dim).cuda()

    def forward(self, wordMatrixOri, wordLeg=None):
        #pdb.set_trace()
        # shorten steps for faster training
        wordMatrix = self.fc1(wordMatrixOri)
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
    if opts.txt_type=='lstmV2':
        txt_encoder =  txtEncoderV2(embedDim, opts.dim_ftr, opts.txt_type)
        return txt_encoder 
    else:
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


class visSeqEncoderV2(nn.Module):
    def __init__(self, embedDim, hidden_dim, seq_Type='lstmV2'):
        super(visSeqEncoderV2, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_type = seq_Type
        if seq_Type =='lstm':
            self.lstm =nn.LSTM(embedDim, hidden_dim, batch_first=True)
        elif seq_Type =='gru':
            self.lstm =nn.GRU(embedDim, hidden_dim, batch_first=True)
        if seq_Type =='lstmV2':
            self.lstm =nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.fc1 = torch.nn.Linear(embedDim, hidden_dim)

        self.hidden = self.init_hidden()
        
    def init_hidden(self, batchSize=10):
        if self.seq_type=='lstm' or self.seq_type=='lstmV2':
            self.hidden=(torch.zeros(1, batchSize, self.hidden_dim).cuda(),
                    torch.zeros(1, batchSize, self.hidden_dim).cuda())
        elif self.seq_type=='gru':
            self.hidden=torch.zeros(1, batchSize, self.hidden_dim).cuda()

    def forward(self, wordMatrixOri, wordLeg=None):
        # shorten steps for faster training
        if self.seq_type =='lstmV2':
            wordMatrix = self.fc1(wordMatrixOri)
        else:
            wordMatrix = wordMatrixOri
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
        pdb.set_trace()
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
    if opts.vis_type == 'lstmV2':
        vis_seq_encoder =  visSeqEncoderV2(embedDim, opts.dim_ftr, opts.vis_type)
        return vis_seq_encoder
    elif opts.vis_type == 'fc':
        vis_avg_encoder = build_vis_fc_encoder(opts)
        return vis_avg_encoder
    elif opts.vis_type == 'vlad_v1':
        vis_vlad_encoder = build_vis_vlad_encoder_v1(opts)
        return vis_vlad_encoder
    elif opts.vis_type == 'avgMIL':
        vis_avg_encoder  = build_vis_fc_encoder(opts)
        return vis_avg_encoder

def build_recontructor(opts):
    if opts.wsMode=='coAttGroundR':
        return txtDecoder(opts.lstm_hidden_size, opts.hdSize, opts.vocaSize)
    else:
        return txtDecoder(opts.hdSize, opts.hdSize, opts.vocaSize)

def build_visDecoder(opts):
    if opts.wsMode=='coAttGroundR':
        return visDecoder(opts.lstm_hidden_size, opts.hdSize, True)
    else:
        return visDecoder(opts.dim_ftr, opts.hdSize, False)


def build_network(opts):
#    pdb.set_trace()
    if opts.wsMode == 'rankTube' or opts.wsMode=='rankFrm': 
        imEncoder= build_vis_seq_encoder(opts)
        wordEncoder = build_txt_encoder(opts)
        wsEncoder = wsEmb(imEncoder, wordEncoder)
    elif opts.wsMode == 'coAtt':
        sst_Obj = SST(opts)
        wsEncoder = wsEmb(sst_Obj, None)
    elif opts.wsMode == 'coAttBi':
        sst_Obj = SSTBi(opts)
        wsEncoder = wsEmb(sst_Obj, None)
    elif opts.wsMode == 'rankGroundR':
        imEncoder= build_vis_seq_encoder(opts)
        wordEncoder = build_txt_encoder(opts)
        wsEncoder = wsEmb(imEncoder, wordEncoder)
        wsEncoder.build_groundR(opts)
    elif opts.wsMode == 'rankGroundRV2':
        imEncoder= build_vis_seq_encoder(opts)
        wordEncoder = build_txt_encoder(opts)
        wsEncoder = wsEmb(imEncoder, wordEncoder)
        wsEncoder.build_groundR(opts)
    elif opts.wsMode == 'coAttGroundR':
        sst_Gr = SSTGroundR(opts)
        wsEncoder = wsEmb(sst_Gr, None)
        wsEncoder.build_groundR(opts)
    elif opts.wsMode == 'coAttV2':
        sst_mul = SSTMul(opts)
        wsEncoder = wsEmb(sst_mul, None)
    elif opts.wsMode == 'coAttV3':
        sst_v3 = SSTV3(opts)
        wsEncoder = wsEmb(sst_v3, None)
    elif opts.wsMode == 'coAttV4':
        sst_v4 = SSTV4(opts)
        wsEncoder = wsEmb(sst_v4, None)
    
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

