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

    more_detailed_flag = True

    for ep in range(opt.stEp, opt.epSize):
        resultList_full = list()
        tBf = time.time() 
        for itr, inputData in enumerate(dataLoader):
            tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list, word_lbl_list, frm_idx_list, bbx_list = inputData
            dataIdx = None
            tmp_bsize = tube_embedding.shape[0]
            imDis = tube_embedding.cuda()
            wordEmb = cap_embedding.cuda()
            wordEmb = wordEmb.view(-1, wordEmb.shape[2], wordEmb.shape[3])
            tAf = time.time()

            if opt.wsMode=='rankFrm':
                imDis = imDis.view(-1, 1, imDis.size(3))
                imFtr, txtFtr = model(imDis, wordEmb, cap_length_list)
                imFtr = imFtr.view(tmp_bsize, -1, opt.dim_ftr)
                txtFtr = txtFtr.view(tmp_bsize, -1, opt.dim_ftr)
#                pdb.set_trace()
                loss = lossEster(imFtr, txtFtr, vd_name_list)
                resultList = evalAcc_frm(imFtr, txtFtr, tubeInfo, indexOri, datasetOri, frm_idx_list, bbx_list, opt.visRsFd+str(ep), False)
                resultList_full +=resultList
            if opt.wsMode =='rankGroundR' or opt.wsMode=='rankGroundRV2':
                imDis = imDis.view(tmp_bsize, -1, imDis.shape[1], imDis.shape[2])
                wordEmb = wordEmb.view(tmp_bsize, -1, wordEmb.shape[1], wordEmb.shape[2])
                logMat, simMM = model(imDis, wordEmb, cap_length_list)
                simMM = simMM.unsqueeze(dim=2).expand(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)
                simMM = simMM.view(tmp_bsize, opt.rpNum, tmp_bsize, opt.capNum)            
                #pdb.set_trace()
                loss = lossEster(logMat, word_lbl_list, simMM, vd_name_list)

            if loss<=0:
                continue
            optimizer.zero_grad()
            loss.backward(retain_graph=True )
            optimizer.step()
            tNf = time.time() 
            if(itr%opt.visIter==0):
                #tAf = time.time()
                logger('Ep: %d, Iter: %d, T1: %3f, T2:%3f, loss: %3f\n' %(ep, itr, (tAf-tBf), (tNf-tAf)/opt.visIter,  float(loss.data.cpu().numpy())))
                tBf = time.time()
                writer.add_scalar('loss', loss.data.cpu()[0], ep*len(datasetOri)+itr*opt.batchSize)
                accSum = 0
                for ele in resultList:
                    recall_k= ele[1]
                    accSum +=recall_k
                logger('Average accuracy on training batch is %3f\n' %(accSum*1.0/len(resultList)))
            tBf = time.time() 

        ## evaluation within an epoch
            if(ep % opt.saveEp==0 and itr==0 and ep >-1):
                checkName = opt.outPre+'_ep_'+str(ep) +'_itr_'+str(itr)+'.pth'
                save_check_point(model.state_dict(), file_name=checkName)
                model.eval()
                resultList = list()
                vIdList = list()
                set_name_ori= opt.set_name
                opt.set_name = 'val'
                frm_num_ori = opt.frm_num 
                opt.frm_num = -1
                opt.no_shuffle_flag =True
                batchSizeOri = opt.batchSize
                opt.batchSize = 1
                dataLoaderEval, datasetEvalOri = build_dataloader(opt) 
                #pdb.set_trace()
                for itr_eval, inputData in enumerate(dataLoaderEval):
                    tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list, word_lbl_list, frm_idx, bbx_list = inputData
                    #pdb.set_trace()
                    dataIdx = None
                    #pdb.set_trace()
                    tmp_bsize = tube_embedding.shape[1]
                    # B*P*T*D
                    imDis = tube_embedding.cuda()
                    wordEmb = cap_embedding.cuda()
                    wordEmb = wordEmb.view(-1, wordEmb.shape[2], wordEmb.shape[3])
                    imDis.requires_grad=False
                    wordEmb.requires_grad=False
                    if opt.wsMode=='rankFrm':
                        imDis = imDis.view(-1, 1, imDis.size(3))
                        imFtr, txtFtr = model(imDis, wordEmb, cap_length_list)
                        imFtr = imFtr.view(tmp_bsize, -1, opt.dim_ftr)
                        txtFtr = txtFtr.view(1, -1, opt.dim_ftr)
        #                pdb.set_trace()
                        resultList += evalAcc_frm_tube(imFtr, txtFtr, tubeInfo, indexOri, datasetEvalOri, frm_idx_list, bbx_list, opt.visRsFd+str(ep), False, more_detailed_flag=more_detailed_flag)
                opt.frm_num = frm_num_ori
                opt.batchSize = batchSizeOri
                accSum = 0
                for ele in resultList:
                    recall_k= ele[1]
                    accSum +=recall_k
                logger('Average accuracy on validation set is %3f\n' %(accSum/len(resultList)))
                writer.add_scalar('Average validation accuracy', accSum/len(resultList), ep*len(datasetOri)+ itr*opt.batchSize)
                out_result_fn = opt.logFd + 'result_val_' +os.path.basename(opt.initmodel).split('.')[0] + '.pk'
                pickledump(out_result_fn, resultList)
                
                #pdb.set_trace()
                model.train()
                opt.set_name = set_name_ori
                opt.no_shuffle_flag =False

        ## evaluation within an epoch
            if(ep % opt.saveEp==0 and itr==0 and ep >0):
                checkName = opt.outPre+'_ep_'+str(ep) +'_itr_'+str(itr)+'.pth'
                save_check_point(model.state_dict(), file_name=checkName)
                model.eval()
                resultList = list()
                vIdList = list()
                set_name_ori= opt.set_name
                opt.set_name = 'test'
                opt.no_shuffle_flag =True
                frm_num_ori = opt.frm_num 
                opt.frm_num = -1
                batchSizeOri = opt.batchSize
                opt.batchSize = 1
                dataLoaderEval, datasetEvalOri = build_dataloader(opt) 
                #pdb.set_trace()
                for itr_eval, inputData in enumerate(dataLoaderEval):
                    tube_embedding, cap_embedding, tubeInfo, indexOri, cap_length_list, vd_name_list, word_lbl_list, frm_idx, bbx_list = inputData
                    #pdb.set_trace()
                    dataIdx = None
                    #pdb.set_trace()
                    tmp_bsize = tube_embedding.shape[1]
                    # B*P*T*D
                    imDis = tube_embedding.cuda()
                    wordEmb = cap_embedding.cuda()
                    wordEmb = wordEmb.view(-1, wordEmb.shape[2], wordEmb.shape[3])
                    imDis.requires_grad=False
                    wordEmb.requires_grad=False
                    if opt.wsMode=='rankFrm':
                        imDis = imDis.view(-1, 1, imDis.size(3))
                        imFtr, txtFtr = model(imDis, wordEmb, cap_length_list)
                        imFtr = imFtr.view(tmp_bsize, -1, opt.dim_ftr)
                        txtFtr = txtFtr.view(1, -1, opt.dim_ftr)
        #                pdb.set_trace()
                        resultList += evalAcc_frm_tube(imFtr, txtFtr, tubeInfo, indexOri, datasetEvalOri, frm_idx_list, bbx_list, opt.visRsFd+str(ep), False, more_detailed_flag=more_detailed_flag)
                opt.frm_num = frm_num_ori
                opt.batchSize = batchSizeOri
                accSum = 0
                for ele in resultList:
                    recall_k= ele[1]
                    accSum +=recall_k
                logger('Average accuracy on testing set is %3f\n' %(accSum/len(resultList)))
                writer.add_scalar('Average testing accuracy', accSum/len(resultList), ep*len(datasetOri)+ itr*opt.batchSize)
                out_result_fn = opt.logFd + 'result_test_' +os.path.basename(opt.initmodel).split('.')[0] + '.pk'
                pickledump(out_result_fn, resultList)
                #pdb.set_trace()
                model.train()
                opt.no_shuffle_flag =False
                break


        accSum = 0
        for ele in resultList_full:
            index, recall_k= ele
            accSum +=recall_k
        writer.add_scalar('Average training accuracy', accSum/len(resultList_full), ep*len(datasetOri)+ itr*opt.batchSize)
