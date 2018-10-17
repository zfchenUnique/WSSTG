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

    if opt.eval_val_flag:
                
        visRsFd = '../data/visResult/actNet/%s_val \n' %(os.path.basename(opt.initmodel))
        model.eval()
        resultList = list()
        vIdList = list()
        opt.set_name = 'val'
        opt.capNum = 5
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
                resultList += evalAcc_actNet(imFtr, txtFtr, tube_info_list, person_list, datasetEvalOri.jpg_folder, visRsFd, False)
            if opt.wsMode =='coAtt':
                simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.view(b_size, opt.rpNum, b_size, opt.capNum)            
                resultList += evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+'/val', True)
        
        accSum = 0
        for ele in resultList:
            index, recall_k= ele
            accSum +=recall_k
        logger('Average accuracy on validation set is %3f\n' %(accSum/len(resultList)))
    
    if opt.eval_test_flag:
        
        visRsFd = '../data/visResult/actNet/%s_test \n' %(os.path.basename(opt.initmodel))
        resultList = list()
        vIdList = list()
        set_name_ori= opt.set_name
        opt.set_name = 'test'
        opt.capNum = 5
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
                resultList += evalAcc_actNet(imFtr, txtFtr, tube_info_list, person_list, datasetEvalOri.jpg_folder, visRsFd, False)
            if opt.wsMode =='coAtt':
                simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.view(b_size, opt.rpNum, b_size, opt.capNum)            
                resultList += evalAcc_actNet_att(simMM, tube_info_list, person_list, datasetOri.jpg_folder, opt.visRsFd+'/test', True)

        accSum = 0
        for ele in resultList:
            index, recall_k= ele
            accSum +=recall_k
        logger('Average accuracy on testing set is %3f\n' %(accSum/len(resultList)))

