from wsParamParser import parse_args
from data.data_loader  import* 
from datasetLoader import *
from modelArc import *
from optimizers import *
from logInfo import logInF
from lossPackage import *
from netUtil import *
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

    for ep in range(opt.epSize):
        
        if(ep % opt.saveEp==0 and ep>0):
        #if(ep % opt.saveEp==0):
            datasetOri.image_samper_set_up(imNum=-1, rpNum=20, capNum=1, maxWordNum=15, trainFlag=False, rndSmpImgFlag=False)
            model.eval()
            resultList = list()
            for itr, inputData in enumerate(dataLoader):
                imDis, wordEmb, lbl, prpList, gtBbxList, frmList, capLbl = inputData
                #pdb.set_trace()
                if opt.gpu:
                    imDis = imDis.cuda()
                    wordEmb = wordEmb.cuda()
                    imDis.requires_grad=False
                    wordEmb.requires_grad=False
                imFtr, txtFtr = model(imDis, wordEmb)
                resultList += evalAcc(imFtr, txtFtr, lbl, prpList, gtBbxList, datasetOri.data['vd'], frmList, capLbl,  datasetOri.data)
            accSum = 0
            for ele in resultList:
                vdName, acc, frmList= ele
                accSum +=acc
            logger('Average accuracy on validation set is %3f\n' %(accSum/len(resultList)))
            model.train()
            dataLoader, datasetOri= build_dataloader(opt) 
            checkName = opt.outPre+str(ep)+'.pth'
            save_check_point(model.state_dict(), file_name=checkName)
        
        
        for itr, inputData in enumerate(dataLoader):
            tBf = time.time() 
            imDis, wordEmb, lbl  = inputData
            #print(lbl)
            if(len(set(lbl))==1):
                continue        # batchsize must be larger than one for siamese loss
            if opt.gpu:
                imDis = imDis.cuda()
                wordEmb = wordEmb.cuda()
            imDis = imDis.view(-1, imDis.shape[2], imDis.shape[3])
            imFtr, txtFtr = model(imDis, wordEmb)
            loss = lossEster(imFtr, txtFtr, lbl)
            if float(loss.data.cpu().numpy())<=0:
                continue
            optimizer.zero_grad()
            loss.backward(retain_graph=True )
            optimizer.step()
            if(itr%opt.visIter==0):
                tAf = time.time()
                logger('Ep: %d, Time: %3f, loss: %3f\n' %(ep, tAf-tBf, float(loss.data.cpu().numpy())))
                tBf = tAf
