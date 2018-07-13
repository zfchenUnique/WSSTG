import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F

class wsEmb(nn.Module):
    def __init__(self, imEncoder, wordEncoder):
        super(wsEmb, self).__init__()
        self._train=True
        self.imEncoder = imEncoder
        self.wordEncoder = wordEncoder
        self._initialize_weights()

    def forward(self, imDis, wordEmb):
        assert len(imDis.size())==3
        assert len(wordEmb.size())==3
        imEnDis = self.imEncoder(imDis)
        wordEnDis = self.wordEncoder(wordEmb)
        assert len(imDis.size())==3
        assert len(wordEmb.size())==3
        imEnDis = F.normalize(imEnDis, p=2, dim=2)
        wordEnDis = F.normalize(wordEnDis, p=2, dim=2)

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
                nn.init.constant_(m.bias, 0)

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

    def forward(self, wordMatrix):
        #pdb.set_trace()
        # shorten steps for faster training
        self.init_hidden(wordMatrix.shape[0])
        lstmOut, self.hidden = self.lstm(wordMatrix, self.hidden)
        #pdb.set_trace()
        return self.hidden[0]

def build_txt_encoder(opts):
    embedDim =300
    txt_encoder =  txtEncoder(embedDim, opts.dim_ftr)
    return txt_encoder 

def build_network(opts):
    imEncoder= build_vis_encoder(opts)
    wordEncoder = build_txt_encoder(opts)
    wsEncoder = wsEmb(imEncoder, wordEncoder)
    if opts.gpu:
        wsEncoder= wsEncoder.cuda()
    if opts.initmodel is not None:
        md_stat = torch.load(opts.initmodel)
        wsEncoder.load_state_dict(md_stat)
    return wsEncoder













