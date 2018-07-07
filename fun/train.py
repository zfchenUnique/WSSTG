from wsParamParser import parse_args
from data.data_loader  import* 
from datasetLoader import *
import pdb

if __name__=='__main__':
    opt = parse_args()
    # build dataloader 
    dataLoader= build_dataloader(opt) 
    # build network 
    model = build_network(opt)
    # build_optimizer
    optimizer = build_opt(opt,  model)

    for itr, inputData in enumerate(dataLoader):
        imDis, wordEmb, lbl  = inputData
        imDis = imDis.view(-1, imDis.shape[2], imDis.shape[3])

    # build loss layer
    lossEster = build_lossEval(opt)
