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
        #if(ep % opt.saveEp==0 and ep<0):
        if(ep % opt.saveEp==0):
            checkName = opt.outPre+'_ep_'+str(ep)+'.pth'
            save_check_point(model.state_dict(), file_name=checkName)
            model.eval()
            resultList = list()
            vIdList = list()
            opt.set_name = 'val'
            dataLoader, datasetOri = build_dataloader(opt) 
            for itr, inputData in enumerate(dataLoader):
                tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list = inputData
                dataIdx = None
                #pdb.set_trace()
                b_size = tube_embedding.shape[0]
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
                    resultList += evalAcc(imFtr, txtFtr, tubeInfo, indexOri, datasetOri, opt.visRsFd+str(ep), True)

            accSum = 0
            for ele in resultList:
                index, recall_k= ele
                accSum +=recall_k
            logger('Average accuracy on validation set is %3f\n' %(accSum/len(resultList)))
            writer.add_scalar('Average accuracy', accSum/len(resultList), ep)
            pdb.set_trace()
            model.train()
            opt.set_name = 'train'
            dataLoader, datasetOri= build_dataloader(opt) 
            checkName = opt.outPre+str(ep)+'.pth'
            save_check_point(model.state_dict(), file_name=checkName)
        
        for itr, inputData in enumerate(dataLoader):
            tBf = time.time() 
            tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list = inputData
            dataIdx = None
            if opt.gpu:
                tmp_bsize = tube_embedding.shape[0]
                imDis = tube_embedding.cuda()
                imDis = imDis.view(-1, imDis.shape[2], imDis.shape[3])
                wordEmb = cap_embedding.cuda()
                wordEmb = wordEmb.view(-1, wordEmb.shape[2], wordEmb.shape[3])
            if opt.wsMode=='rankTube':
#                pdb.set_trace()
                imFtr, txtFtr = model(imDis, wordEmb, cap_length_list)
                imFtr = imFtr.view(tmp_bsize, -1, opt.dim_ftr)
                txtFtr = txtFtr.view(tmp_bsize, -1, opt.dim_ftr)
#                pdb.set_trace()
                loss = lossEster(imFtr, txtFtr, vd_name_list)

            if float(loss.data.cpu().numpy())<=0:
                continue
            optimizer.zero_grad()
            loss.backward(retain_graph=True )
            tL2 = time.time()
            optimizer.step()
            if(itr%opt.visIter==0):
                tAf = time.time()
                logger('Ep: %d, Iter: %d, Time: %3f, loss: %3f\n' %(ep, itr, tAf-tBf, float(loss.data.cpu().numpy())))
                tBf = tAf
                writer.add_scalar('loss', loss.data.cpu()[0], ep*len(dataLoader)+itr)
        dataLoader, datasetOri= build_dataloader(opt) 
        checkName = opt.outPre+str(ep)+'.pth'
        save_check_point(model.state_dict(), file_name=checkName)
