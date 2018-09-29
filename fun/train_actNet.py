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
    for ep in range(opt.stEp, opt.epSize):
        resultList_full = list()
        tBf = time.time()
        for itr, inputData in enumerate(dataLoader):
            tube_embedding, cap_embedding, tube_info_list, person_list, cap_length_list, shot_list = inputData
            tDf = time.time()
            #pdb.set_trace()
            dataIdx = None
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
                loss = lossEster(imFtr, txtFtr, shot_list)
                resultList = evalAcc_actNet(imFtr, txtFtr, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False)
                resultList_full +=resultList

            if loss<=0:
                continue
            optimizer.zero_grad()
            loss.backward(retain_graph=True )
            tL2 = time.time()
            optimizer.step()
            if(itr%opt.visIter==0):
                tAf = time.time()
                logger('Ep: %d, Iter: %d, T1: %3f, T2:%3f, loss: %3f\n' %(ep, itr, (tDf-tBf)/opt.visIter, (tAf-tBf)/opt.visIter,  float(loss.data.cpu().numpy())))
                writer.add_scalar('loss', loss.data.cpu()[0], ep*len(datasetOri)+itr*opt.batchSize)
                accSum = 0
                for ele in resultList:
                    index, recall_k= ele
                    accSum +=recall_k
                logger('Average accuracy on training batch is %3f\n' %(accSum/len(resultList)))
                tBf = time.time()
        ## evaluation within an epoch
            if(ep % opt.saveEp==0 and itr==200 and itr>0):
                checkName = opt.outPre+'_ep_'+str(ep) +'_itr_'+str(itr)+'.pth'
                save_check_point(model.state_dict(), file_name=checkName)
                model.eval()
                resultList = list()
                vIdList = list()
                set_name_ori= opt.set_name
                opt.set_name = 'val'
                dataLoaderEval, datasetEvalOri = build_dataloader(opt) 
                #pdb.set_trace()
                for itr_eval, inputData in enumerate(dataLoaderEval):
                    tube_embedding, cap_embedding, tube_info_list, person_list, cap_length_list, shot_list = inputData
                    #pdb.set_trace()
                    dataIdx = None
                    #pdb.set_trace()
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
                        resultList += evalAcc_actNet(imFtr, txtFtr, tube_info_list, person_list, datasetEvalOri.jpg_folder, opt.visRsFd+str(ep), False)

                accSum = 0
                for ele in resultList:
                    index, recall_k= ele
                    accSum +=recall_k
                logger('Average accuracy on validation set is %3f\n' %(accSum/len(resultList)))
                writer.add_scalar('Average testing accuracy', accSum/len(resultList), ep*len(datasetOri)+ itr*opt.batchSize)
                #pdb.set_trace()
                model.train()
                opt.set_name = set_name_ori
                
        accSum = 0
        for ele in resultList_full:
            index, recall_k= ele
            accSum +=recall_k
        writer.add_scalar('Average training accuracy', accSum/len(resultList_full), ep*len(datasetOri)+ itr*opt.batchSize)
        checkName = opt.outPre+'_ep_'+str(ep) +'_itr_'+str(itr)+'.pth'
        save_check_point(model.state_dict(), file_name=checkName)
                #dataLoader, datasetOri= build_dataloader(opt) 

