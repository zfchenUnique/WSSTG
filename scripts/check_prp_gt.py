import sys
import numpy
sys.path.append('..')
from util.mytoolbox import *
import pdb

if __name__=='__main__':
    annFn ='../data/annForDb_otbV2.pd'
    prpFd ='/disk2/zfchen/data/otbRpn/'
    isTrainSet=True

    annDict =  pickleload(annFn)
    if isTrainSet:
        bbxListDict=annDict['train_bbx_list']
        vidList = annDict['trainName']
        frmListDict=annDict['trainImg'] 
    else:
        bbxListDict=annDict['test_bbx_list']
        vidList = annDict['testName']
        frmListDict=annDict['testImg']

    for i, vidSet in enumerate(vidList):
        print('%d %s' %(i, vidSet))
        bbxNum = len(bbxListDict[i])
        frmNum = len(frmListDict[i])
        if(bbxNum!=frmNum):
            print('%d %s' %(i, vidSet))
            pdb.set_trace()  





    
