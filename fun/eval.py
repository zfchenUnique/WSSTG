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
    for ep in range(opt.stEp, opt.epSize):
        #if ep==0:
        #    continue
        for itr_eval in range(0, len(dataLoader), 200):
            if ep==0 and itr_eval<=200:
                continue
            checkName = opt.outPre+'_ep_'+str(ep) +'_itr_'+str(itr_eval)+'.pth'
            print(checkName)
            #pdb.set_trace()
            md_stat = torch.load(checkName)
            model.load_state_dict(md_stat)
            model.eval()
            resultList = list()
            vIdList = list()
            set_name_ori= opt.set_name
            #opt.set_name = 'train'
            dataLoader, datasetOri = build_dataloader(opt) 
            for itr, inputData in enumerate(dataLoader):
               tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list = inputData
               b_size = tube_embedding.shape[0]
               # B*P*T*D
               imDis = tube_embedding.cuda()
               imDis = imDis.view(-1, imDis.shape[2], imDis.shape[3])
               wordEmb = cap_embedding.cuda()
               wordEmb = wordEmb.view(-1, wordEmb.shape[2], wordEmb.shape[3])
               imDis.requires_grad=False
               wordEmb.requires_grad=False
               if opt.wsMode=='rankTube':
                   imFtr, txtFtr = model(imDis, wordEmb, cap_length_list)
                   imFtr = imFtr.view(b_size, -1, opt.dim_ftr)
                   txtFtr = txtFtr.view(b_size, -1, opt.dim_ftr)
                   resultList += evalAcc(imFtr, txtFtr, tubeInfo, indexOri, datasetOri, opt.visRsFd+str(ep), False)

            accSum = 0
            for ele in resultList:
               index, recall_k= ele
               accSum +=recall_k
            logger('Average accuracy on %s  set is %3f\n' %(set_name_ori, accSum/len(resultList)))
            writer.add_scalar('Average accuracy', accSum/len(resultList), ep*len(dataLoader)+ itr_eval)

