from wsParamParser import parse_args
from data.data_loader  import* 
from datasetLoader import *
from modelArc import *
from optimizers import *
from logInfo import logInF
from lossPackage import *
from netUtil import *
from tensorboardX import SummaryWriter
import time

import pdb

if __name__=='__main__':
    opt = parse_args()
    # build dataloader
    dataLoader, datasetOri= build_dataloader(opt) 
    # build network 
    model = build_network(opt)
    # build_optimizer
    optimizer = build_opt(opt,  model)
    # build loss layer
    lossEster = build_lossEval(opt)
    # build logger
    logger = logInF(opt.logFd)
    writer = SummaryWriter(opt.logFdTx+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    #pdb.set_trace()

