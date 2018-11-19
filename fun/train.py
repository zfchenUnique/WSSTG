import numpy 
import torch
import random
def random_seeding(seed_value, use_cuda):
    numpy.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value)
    if use_cuda: 
        torch.cuda.manual_seed_all(seed_value) # gpu vars

seed_value = 4
random_seeding(seed_value, True)

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
        resultList_full = list()
        tBf = time.time() 
        for itr, inputData in enumerate(dataLoader):
            tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list, word_lbl_list = inputData
            #pdb.set_trace()
            dataIdx = None
            tmp_bsize = tube_embedding.shape[0]
            imDis = tube_embedding.cuda()
            imDis = imDis.view(-1, imDis.shape[2], imDis.shape[3])
            wordEmb = cap_embedding.cuda()
            wordEmb = wordEmb.view(-1, wordEmb.shape[2], wordEmb.shape[3])
            tAf = time.time()

            #pdb.set_trace()
            if opt.wsMode=='rankTube':
                imFtr, txtFtr = model(imDis, wordEmb, cap_length_list)
                imFtr = imFtr.view(tmp_bsize, -1, opt.dim_ftr)
                txtFtr = txtFtr.view(tmp_bsize, -1, opt.dim_ftr)
#                pdb.set_trace()
                loss = lossEster(imFtr, txtFtr, vd_name_list)
                resultList = evalAcc(imFtr, txtFtr, tubeInfo, indexOri, datasetOri, opt.visRsFd+str(ep), False)
                resultList_full +=resultList
            if opt.wsMode =='coAtt' or opt.wsMode =='coAttV2' or opt.wsMode=='coAttV3' or opt.wsMode=='coAttV4':
                simMM = model(imDis, wordEmb, cap_length_list)
#                pdb.set_trace()
                simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                loss = lossEster(simMM=simMM, lblList =vd_name_list)
                resultList = evalAcc_att(simMM, tubeInfo, indexOri, datasetOri, opt.visRsFd+str(ep), False)
                resultList_full +=resultList
            if opt.wsMode =='rankGroundR' or opt.wsMode=='rankGroundRV2':
                imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                logMat, simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                #pdb.set_trace()
                loss = lossEster(logMat, word_lbl_list, simMM, vd_name_list)
                resultList = evalAcc_att(simMM, tubeInfo, indexOri, datasetOri, opt.visRsFd+str(ep), False)
                resultList_full +=resultList

            if opt.wsMode =='coAttGroundR':
                #pdb.set_trace()
                imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                logMat, simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                #pdb.set_trace()
                print(torch.max(simMM))
                print(torch.min(simMM))
                loss = lossEster(logMat, word_lbl_list, simMM, vd_name_list)
                resultList = evalAcc_att(simMM, tubeInfo, indexOri, datasetOri, opt.visRsFd+str(ep), False)
                resultList_full +=resultList

            if loss<=0:
                continue
            optimizer.zero_grad()
            loss.backward(retain_graph=True )
            optimizer.step()
            optimizer.zero_grad()
            tNf = time.time() 
            if(itr%opt.visIter==0):
                #tAf = time.time()
                logger('Ep: %d, Iter: %d, T1: %3f, T2:%3f, loss: %3f\n' %(ep, itr, (tAf-tBf), (tNf-tAf)/opt.visIter,  float(loss.data.cpu().numpy())))
                tBf = time.time()
                writer.add_scalar('loss', loss.data.cpu()[0], ep*len(datasetOri)+itr*opt.batchSize)
                accSum = 0
                for ele in resultList:
                    index, recall_k= ele
                    accSum +=recall_k
                logger('Average accuracy on training batch is %3f\n' %(accSum/len(resultList)))
            tBf = time.time() 

        ## evaluation within an epoch
            if(ep % opt.saveEp==0 and itr==0 and ep >0):
                checkName = opt.outPre+'_ep_'+str(ep) +'_itr_'+str(itr)+'.pth'
                save_check_point(model.state_dict(), file_name=checkName)
                model.eval()
                resultList = list()
                vIdList = list()
                set_name_ori= opt.set_name
                opt.set_name = 'val'
                batchSizeOri = opt.batchSize
                opt.batchSize = 8
                dataLoaderEval, datasetEvalOri = build_dataloader(opt) 
                opt.batchSize = batchSizeOri 
                #pdb.set_trace()
                for itr_eval, inputData in enumerate(dataLoaderEval):
                    tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list, word_lbl_list = inputData
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
                        resultList += evalAcc(imFtr, txtFtr, tubeInfo, indexOri, datasetEvalOri, opt.visRsFd+str(ep), False)
                    if opt.wsMode =='coAtt' or opt.wsMode =='coAttV2' or opt.wsMode=='coAttV3' or opt.wsMode=='coAttV4':
                        simMM = model(imDis, wordEmb, cap_length_list)
                        simMM = simMM.view(b_size, opt.rpNum, b_size)            
                        resultList += evalAcc_att(simMM, tubeInfo, indexOri, datasetEvalOri, opt.visRsFd+str(ep), False)
                    if opt.wsMode =='rankGroundR' or opt.wsMode=='rankGroundRV2':
                        tmp_bsize = b_size
                        imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                        wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                        logMat, simMM = model(imDis, wordEmb, cap_length_list)
                        simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                        simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                        resultList += evalAcc_att(simMM, tubeInfo, indexOri, datasetEvalOri, opt.visRsFd+str(ep), False)
                    if opt.wsMode =='coAttGroundR':
                        #pdb.set_trace()
                        tmp_bsize = b_size
                        imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                        wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                        logMat, simMM = model(imDis, wordEmb, cap_length_list)
                        simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                        simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                        resultList += evalAcc_att(simMM, tubeInfo, indexOri, datasetEvalOri, opt.visRsFd+str(ep), False)

                accSum = 0
                for ele in resultList:
                    index, recall_k= ele
                    accSum +=recall_k
                logger('Average accuracy on validation set is %3f\n' %(accSum/len(resultList)))
                writer.add_scalar('Average validation accuracy', accSum/len(resultList), ep*len(datasetOri)+ itr*opt.batchSize)
                #pdb.set_trace()
                model.train()
                opt.set_name = set_name_ori

            if(ep % opt.saveEp==0 and itr==0 and ep > 0):
                model.eval()
                resultList = list()
                vIdList = list()
                set_name_ori= opt.set_name
                opt.set_name = 'test'
                batchSizeOri = opt.batchSize
                opt.batchSize = 8
                dataLoaderEval, datasetEvalOri = build_dataloader(opt) 
                opt.batchSize = batchSizeOri 
                #pdb.set_trace()
                for itr_eval, inputData in enumerate(dataLoaderEval):
                    tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list, word_lbl_list = inputData
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
                        resultList += evalAcc(imFtr, txtFtr, tubeInfo, indexOri, datasetEvalOri, opt.visRsFd+str(ep), False)
                    if opt.wsMode =='coAtt' or opt.wsMode =='coAttV2' or opt.wsMode=='coAttV3' or opt.wsMode=='coAttV4':
                        simMM = model(imDis, wordEmb, cap_length_list)
                        simMM = simMM.view(b_size, opt.rpNum, b_size)            
                        resultList += evalAcc_att(simMM, tubeInfo, indexOri, datasetEvalOri, opt.visRsFd+str(ep), False)
                    if opt.wsMode =='rankGroundR' or opt.wsMode=='rankGroundRV2':
                        tmp_bsize = b_size
                        imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                        wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                        logMat, simMM = model(imDis, wordEmb, cap_length_list)
                        simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                        simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                        resultList += evalAcc_att(simMM, tubeInfo, indexOri, datasetEvalOri, opt.visRsFd+str(ep), False)
                    if opt.wsMode =='coAttGroundR':
                        #pdb.set_trace()
                        tmp_bsize = b_size
                        imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                        wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                        logMat, simMM = model(imDis, wordEmb, cap_length_list)
                        simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                        simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                        resultList += evalAcc_att(simMM, tubeInfo, indexOri, datasetEvalOri, opt.visRsFd+str(ep), False)


                accSum = 0
                for ele in resultList:
                    index, recall_k= ele
                    accSum +=recall_k
                logger('Average accuracy on testing set is %3f\n' %(accSum/len(resultList)))
                writer.add_scalar('Average testing accuracy', accSum/len(resultList), ep*len(datasetOri)+ itr*opt.batchSize)
                #pdb.set_trace()
                model.train()
                opt.set_name = set_name_ori
                
        accSum = 0
        for ele in resultList_full:
            index, recall_k= ele
            accSum +=recall_k
        writer.add_scalar('Average training accuracy', accSum/len(resultList_full), ep*len(datasetOri)+ itr*opt.batchSize)
