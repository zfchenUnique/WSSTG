import torch
import pdb
class wsEmb(nn.Module):
    def __init__(self, imEncoder, wordEncoder):
        super(wsEmb, self).__init__()
        self._train=True
        self.imEncoder = imEncoder
        self.wordEncoder = wordEncoder
        self._initialize_weights()

    def forward(self, imDis, wordEmb):
        imEnDis = self.imEncoder(imDis)
        wordEnDis = self.wordEncoder(wordEmb)


    def _initialize_weights(self):
        #pdb.set_trace()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.01)
                nn.init.constant(m.bias, 0)

def build_vis_encoder(opts):
    inputDim = 2048
    visNet = torch.nn.Sequential(
            torch.nn.Linear(inputDim, inputDim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(inputDim, opts.dim_ftr),
            )
    return visNet

def build_txt_encoder(opts):



    


def build_network(opts):
    imEncoder= build_vis_encoder(opts)
    wordEncoder = build_txt_encoder(opts)
    wsEncoder = wsEmb(imEncoder, wordEncoder)
    return wsEncoder













