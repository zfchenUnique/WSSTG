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
    # build logger
    logger = logInF(opt.logFd)
    ep = 0
    thre_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #thre_list = [0.5]
    #thre_list = [0.5, 0.7]
    acc_list = list()
    for thre in thre_list:
        acc_list.append([])
    
    if opt.eval_val_flag:
        model.eval()
        resultList = list()
        vIdList = list()
        cap_num_ori = opt.capNum
        opt.set_name = 'val'
        opt.capNum = 5
        dataLoaderEval, datasetEvalOri = build_dataloader(opt) 
        for itr_eval, inputData in enumerate(dataLoaderEval):
            tube_embedding, cap_embedding, tube_info_list, person_list, cap_length_list, shot_list, word_lbl_list = inputData
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
                for i, thre in enumerate(thre_list):
                    acc_list[i] += evalAcc_actNet(imFtr, txtFtr, tube_info_list, person_list, datasetEvalOri.jpg_folder, opt.visRsFd+str(ep), False, thre_list=[thre])

            if opt.wsMode =='coAtt':
                simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.view(b_size, opt.rpNum, b_size, opt.capNum)            
                for i, thre in enumerate(thre_list):
                    acc_list[i] += evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False, thre_list=[thre])
            
            if opt.wsMode =='rankGroundR' or opt.wsMode=='rankGroundRV2':
                tmp_bsize = b_size
                imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                logMat, simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                #pdb.set_trace()
                for i, thre in enumerate(thre_list):
                    acc_list[i] += evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False, thre_list=[thre])
        
        for i, thre in enumerate(thre_list):
            resultList = acc_list[i]
            accSum = 0
            for ele in resultList:
                index, recall_k= ele
                accSum +=recall_k
            logger('thre @ %f, Average accuracy on validation set is %3f\n' %(thre, accSum/len(resultList)))

    acc_list = list()
    for thre in thre_list:
        acc_list.append([])
        # testing on testing set
    if opt.eval_test_flag:
        model.eval()
        resultList = list()
        vIdList = list()
        opt.set_name = 'test'
        opt.capNum = 5
        if opt.wsMode =='coAtt':
            opt.capNum = 1
        dataLoaderEval, datasetEvalOri = build_dataloader(opt) 
        #pdb.set_trace()
        for itr_eval, inputData in enumerate(dataLoaderEval):
            tube_embedding, cap_embedding, tube_info_list, person_list, cap_length_list, shot_list, word_lbl_list = inputData
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
                for i, thre in enumerate(thre_list):
                    acc_list[i] += evalAcc_actNet(imFtr, txtFtr, tube_info_list, person_list, datasetEvalOri.jpg_folder, opt.visRsFd+str(ep), False, thre_list=[thre])

            if opt.wsMode =='coAtt':
                simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.view(b_size, opt.rpNum, b_size, opt.capNum)            
                for i, thre in enumerate(thre_list):
                    acc_list[i] += evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False, thre_list=[thre])
            if opt.wsMode =='rankGroundR' or opt.wsMode=='rankGroundRV2':
                tmp_bsize = b_size
                imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                logMat, simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                for i, thre in enumerate(thre_list):
                    acc_list[i] += evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+str(ep), False, thre_list=[thre])
        
        for i, thre in enumerate(thre_list):
            resultList = acc_list[i]
            accSum = 0
            for ele in resultList:
                index, recall_k= ele
                accSum +=recall_k
            logger('thre @ %f, Average accuracy on testing set is %3f\n' %(thre, accSum/len(resultList)))
        
