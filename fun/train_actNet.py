import numpy 
import torch
import random
def random_seeding(seed_value, use_cuda):
    numpy.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value)
    if use_cuda: 
        torch.cuda.manual_seed_all(seed_value) # gpu vars

seed_value = 1
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
    for ep in range(opt.stEp, opt.epSize):
        resultList_full = list()
        tBf = time.time()
        for itr, inputData in enumerate(dataLoader):
            tube_embedding, cap_embedding, tube_info_list, person_list, cap_length_list, shot_list = inputData
            tDf = time.time()
            #print(shot_list)
            #print(cap_length_list)
            #pdb.set_trace()
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
                #tDf2 = time.time()
                loss = lossEster(imFtr, txtFtr, shot_list)
                #tDf3 = time.time()
                #print(tDf3-tDf2)
            if opt.wsMode =='coAtt':
                simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                loss = lossEster(simMM=simMM, lblList =shot_list)

            if loss<=0:
                continue
            optimizer.zero_grad()
            loss.backward(retain_graph=True )
            tL2 = time.time()
            tAf = time.time()
            optimizer.step()
            if(itr%opt.visIter==0):
                logger('Ep: %d, Iter: %d, T1: %3f, T2:%3f, loss: %3f\n' %(ep, itr, (tDf-tBf), (tAf-tBf),  float(loss.data.cpu().numpy())))
                writer.add_scalar('loss', loss.data.cpu()[0], ep*len(datasetOri)+itr*opt.batchSize)
                resultList = list()
                if opt.wsMode=='rankTube':
                    resultList = evalAcc_actNet(imFtr, txtFtr, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False)
                elif opt.wsMode=='coAtt':
                    resultList = evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False)
                resultList_full +=resultList
                accSum = 0
                for ele in resultList:
                    #continue
                    index, recall_k= ele
                    accSum +=recall_k
                logger('Average accuracy on training batch is %3f\n' %(accSum/(len(resultList)+0.000001)))
        ## evaluation within an epoch
            tBf = time.time()
            if(ep % opt.saveEp==0 and itr==0 and ep>0):
#                pdb.set_trace()
                checkName = opt.outPre+'_ep_'+str(ep) +'_itr_'+str(itr)+'.pth'
                save_check_point(model.state_dict(), file_name=checkName)
                model.eval()
                resultList = list()
                vIdList = list()
                set_name_ori= opt.set_name
                cap_num_ori = opt.capNum
                opt.set_name = 'val'
                opt.capNum = 5
                if opt.wsMode =='coAtt':
                    opt.capNum = 1
                dataLoaderEval, datasetEvalOri = build_dataloader(opt) 
                #pdb.set_trace()
                for itr_eval, inputData in enumerate(dataLoaderEval):
                    tube_embedding, cap_embedding, tube_info_list, person_list, cap_length_list, shot_list = inputData
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

                    if opt.wsMode =='coAtt':
                        simMM = model(imDis, wordEmb, cap_length_list)
                        simMM = simMM.view(b_size, opt.rpNum, b_size, opt.capNum)            
                        resultList += evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False)
                
                accSum = 0
                for ele in resultList:
                    index, recall_k= ele
                    accSum +=recall_k
                logger('Average accuracy on validation set is %3f\n' %(accSum/len(resultList)))
                writer.add_scalar('Average validation accuracy', accSum/len(resultList), ep*len(datasetOri)+ itr*opt.batchSize)
                #pdb.set_trace()
                # testing on testing set
                resultList = list()
                vIdList = list()
                set_name_ori= opt.set_name
                opt.set_name = 'test'
                opt.capNum = 5
                if opt.wsMode =='coAtt':
                    opt.capNum = 1          # for memory issue
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

                    if opt.wsMode =='coAtt':
                        simMM = model(imDis, wordEmb, cap_length_list)
                        simMM = simMM.view(b_size, opt.rpNum, b_size, opt.capNum)            
                        resultList += evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False)
                accSum = 0
                for ele in resultList:
                    index, recall_k= ele
                    accSum +=recall_k
                logger('Average accuracy on testing set is %3f\n' %(accSum/len(resultList)))
                writer.add_scalar('Average testing accuracy', accSum/len(resultList), ep*len(datasetOri)+ itr*opt.batchSize)
                
                
                model.train()
                opt.set_name = set_name_ori
                opt.capNum = cap_num_ori
                
        accSum = 0
        for ele in resultList_full:
            #continue
            index, recall_k= ele
            accSum +=recall_k
        writer.add_scalar('Average training accuracy', accSum/(len(resultList_full)+0.000001), ep*len(datasetOri)+ itr*opt.batchSize)
        checkName = opt.outPre+'_ep_'+str(ep) +'_itr_'+str(itr)+'.pth'
        save_check_point(model.state_dict(), file_name=checkName)
                #dataLoader, datasetOri= build_dataloader(opt) 

