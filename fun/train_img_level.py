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
#        if(ep % opt.saveEp==0 and ep>0):
        if(ep % opt.saveEp==0):
            checkName = opt.outPre+str(ep)+'.pth'
            save_check_point(model.state_dict(), file_name=checkName)
            datasetOri.image_samper_set_up(imNum=-1, rpNum=20, capNum=1, maxWordNum=opt.maxWL, trainFlag=False, rndSmpImgFlag=False, conFrmNum= opt.conFrmNum, conSecFlag=opt.conSecFlag)
            model.eval()
            resultList = list()
            vIdList = list()
#            pdb.set_trace()
            for itr, inputData in enumerate(dataLoader):
                imDis, wordEmb, lbl, prpList, gtBbxList, frmList, capLbl, wordLbl, capLength = inputData
                dataIdx = None
                if opt.isParal:
                    dataIdx = torch.arange(0, len(capLength))
                #pdb.set_trace()
                if opt.gpu:
                    imDis = imDis.cuda()
                    wordEmb = wordEmb.cuda()
                    imDis.requires_grad=False
                    wordEmb.requires_grad=False
                if opt.wsMode=='rank':
                    imFtr, txtFtr = model(imDis, wordEmb, capLength, dataIdx=dataIdx)
#                    pdb.set_trace()
                    resultList += evalAcc(imFtr, txtFtr, lbl, prpList, gtBbxList, datasetOri.data['vd'], frmList, capLbl,  datasetOri.data, opt.visRsFd+str(ep))
                elif opt.wsMode=='groundR':
                    logMat, rpSS = model(imDis, wordEmb, capLength, frmList)
                    resultList += evalAccGroundR(rpSS, logMat, lbl, prpList, gtBbxList, datasetOri.data['vd'], frmList, capLbl,  datasetOri.data)
                elif opt.wsMode == 'graphSpRank' or opt.wsMode == 'graphSpRankC':
                    imDis = imDis.view(-1, opt.conFrmNum, imDis.shape[1], imDis.shape[2])
                    #imFtr, txtFtr = model(imDis, wordEmb, capLength, rpList=prpList)
                    imFtr, txtFtr, aff_softmax, aff_scale, aff_weight = model(imDis, wordEmb, capLength, rpListFull=prpList, dataIdx=dataIdx)
                    #pdb.set_trace()
                    visRelations(datasetOri.data, lbl, prpList, frmList, gtBbxList, capLbl, opt.visRsFd+'spRe'+str(ep), aff_softmax, aff_scale, aff_weight)
                    if opt.keepKeyFrameOnly:
                        imFtr, prpList, frmList= keepKeyFrmForTest(imFtr,  prpList, frmList)
                    #pdb.set_trace()
                    resultList += evalAcc(imFtr, txtFtr, lbl, prpList, gtBbxList, datasetOri.data['vd'], frmList, capLbl,  datasetOri.data, opt.visRsFd+str(ep))


            accSum = 0
            for ele in resultList:
                vdName, acc, frmList= ele
                accSum +=acc
            logger('Average accuracy on validation set is %3f\n' %(accSum/len(resultList)))
            writer.add_scalar('Average accuracy', accSum/len(resultList), ep)
            model.train()
            dataLoader, datasetOri= build_dataloader(opt) 
            checkName = opt.outPre+str(ep)+'.pth'
            save_check_point(model.state_dict(), file_name=checkName)
        
        for itr, inputData in enumerate(dataLoader):
            tBf = time.time() 
            imDis, wordEmb, lbl, prpList, gtBbxList, frmList, capLbl, wordLbl, capLength = inputData
            #pdb.set_trace()
            dataIdx = None
            if opt.isParal:
                dataIdx = torch.arange(0, len(capLength))

            if opt.gpu:
                imDis = imDis.cuda()
                wordEmb = wordEmb.cuda()
            if opt.wsMode=='rank':
                imFtr, txtFtr = model(imDis, wordEmb, capLengt, dataIdx = dataIdx)
                loss = lossEster(imFtr, txtFtr, lbl)
            elif opt.wsMode =='groundR':
                logMat, rpSS = model(imDis, wordEmb, capLength)
                loss = lossEster(logMat, wordLbl)
            elif opt.wsMode == 'graphSpRank' or opt.wsMode == 'graphSpRankC':
                imDis = imDis.view(-1, opt.conFrmNum, imDis.shape[1], imDis.shape[2])
                imFtr, txtFtr, aff_softmax, aff_scale, aff_weight = model(imDis, wordEmb, capLength, rpListFull=prpList, dataIdx=dataIdx)
                loss = lossEster(imFtr, txtFtr, lbl)

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
