import os
import sys
sys.path.append('../')
sys.path.append('../util')
from util.mytoolbox import get_specific_file_list_from_fd, textread, split_carefully, parse_mul_num_lines, pickleload, pickledump, get_list_dir
import pdb
import h5py
import csv

def build_idx_from_list(vdList):
    vdListU = list(set(vdList))
    vd2idx = {}
    idx2vd = {}
    for i, ele in enumerate(vdList):
        vd2idx[ele] = i
        idx2vd[i] =  ele
    return vd2idx, idx2vd

def get_word_list(full_name_list):
    capList =[]
    for filePath in full_name_list: 
        lines = textread(filePath)
        subCapList=[]
        for line in lines:
            wordList=line.split(line, ' ')
            subCapList.append(wordList.lower())
        capList.append(subCapList)
    return capList

def fix_otb_frm_bbxList(annFn, outFn):
    annDict = pickleload(annFn)
   
    isTrainSet = True
    if isTrainSet:
        bbxListDict=annDict['train_bbx_list']
        vidList = annDict['trainName']
        frmListDict=annDict['trainImg'] 
    else:
        bbxListDict=annDict['test_bbx_list']
        vidList = annDict['testName']
        frmListDict=annDict['testImg']

    for i, vidName in enumerate(vidList):
        frmList = frmListDict[i]
        bbxList = bbxListDict[i]
        frmList.sort()
        if(vidName=='David'):
            annDict['trainImg'][i] = frmList[299:]

    isTrainSet = False
    if isTrainSet:
        bbxListDict=annDict['train_bbx_list']
        vidList = annDict['trainName']
        frmListDict=annDict['trainImg'] 
    else:
        bbxListDict=annDict['test_bbx_list']
        vidList = annDict['testName']
        frmListDict=annDict['testImg']

    for i, vidName in enumerate(vidList):
        frmList = frmListDict[i]
        bbxList = bbxListDict[i]
        frmList.sort()
        if(vidName=='David'):
            annDict['testImg'][i] = frmList[299:]

    pickledump(outFn, annDict)
    return annDict

# parse OTB 99
def get_otb_data(inFd):
    test_text_fd = inFd + '/OTB_query_test'     
    train_text_fd = inFd + '/OTB_query_train'     
    video_fd = inFd + '/OTB_videos'
    test_name_list = get_specific_file_list_from_fd(test_text_fd, '.txt') 
    train_name_list = get_specific_file_list_from_fd(train_text_fd, '.txt')
    test_name_listFull = get_specific_file_list_from_fd(test_text_fd, '.txt', False) 
    train_name_listFull = get_specific_file_list_from_fd(train_text_fd, '.txt', False)
    
    test_cap_list = get_word_list(test_name_listFull) 
    train_cap_list = get_word_list(train_name_listFull)

    test_im_list =[]
    test_bbx_list =[]
    for i, vdName  in enumerate(test_name_list):
        full_vd_path = video_fd + '/'+ vdName
        vd_frame_path = full_vd_path + '/img'
        imgList = get_specific_file_list_from_fd(vd_frame_path, '.jpg', True)
        test_im_list.append(imgList)

        gtBoxFn = full_vd_path +'/groundtruth_rect.txt'
        bbxList= parse_mul_num_lines(gtBoxFn)        
        test_bbx_list.append(bbxList)


    train_im_list =[]
    train_bbx_list =[]
    for i, vdName  in enumerate(train_name_list):
        full_vd_path = video_fd + '/'+ vdName
        vd_frame_path = full_vd_path + '/img'
        imgList = get_specific_file_list_from_fd(vd_frame_path, '.jpg')
        train_im_list.append(imgList)

        gtBoxFn = full_vd_path +'/groundtruth_rect.txt'
        bbxList= parse_mul_num_lines(gtBoxFn)   
        train_bbx_list.append(bbxList)

    otb_info_raw= {'trainName': train_name_list, 'testName': test_name_list, 'trainCap': train_cap_list, 'testCap': test_cap_list, 'trainImg': train_im_list, 'testImg': test_im_list, 'train_bbx_list': train_bbx_list, 'test_bbx_list': test_bbx_list}
    return otb_info_raw

def otbPCK2List(pckFn):
    otbDict = pickleload(pckFn)
    testVdList= otbDict['testName']
    testImList = otbDict['testImg']
    trainVdList= otbDict['trainName']
    trainImList = otbDict['trainImg']
    imgList = list()
    for i, vdName in enumerate(testVdList):
        vd_frame_path = vdName + '/img'
        for j, imName in enumerate(testImList[i]):
            imNameFull = vd_frame_path+'/'+imName+'.jpg'
            imgList.append(imNameFull)

    for i, vdName in enumerate(trainVdList):
        vd_frame_path = vdName + '/img'
        for j, imName in enumerate(trainImList[i]):
            imNameFull = vd_frame_path+'/'+imName+'.jpg'
            imgList.append(imNameFull)
    
    return imgList

def a2dSetParser(annFn, annFd, annIgListFn, annFnOri):
    fLineList  = textread(annFn)
    videoList = list()
    capList = list()
    insList = list()
    bbxList = list()
    frmNameList  = list()
    splitDict = {}

    with open(annFnOri, 'rb') as csvFile:
        lines = csv.reader(csvFile)
        for line in lines:
            eleSegs = split_carefully(line, ',')
            splitDict[eleSegs[0][0]]=int(eleSegs[0][-1])

    for i, line in enumerate(fLineList):
        if(i<=0):
            continue
        splitSegs= split_carefully(line, ',')
        annFdSub = annFd + '/' + splitSegs[0]
        annNameList = get_specific_file_list_from_fd(annFdSub, '.h5') 
        tmpFrmList = list()
        tmpBbxList = list()
        #pdb.set_trace()
        insIdx = int(splitSegs[1])
        print(splitSegs[2])
        print('%s %d %d\n' %(splitSegs[0], i, insIdx))
        for ii, annName  in enumerate(annNameList):
            annSubFullPath = annFdSub + '/' +  annName +'.h5'
            annIns = h5py.File(annSubFullPath)
            tmpInsList = list(annIns['instance'][:])
            if(insIdx in tmpInsList):
                tmpFrmList.append(annName)
                bxIdx = tmpInsList.index(insIdx)
                tmpBbxList.append(annIns['reBBox'][:, bxIdx])
        frmNameList.append(tmpFrmList)
        bbxList.append(tmpBbxList)
        videoList.append(splitSegs[0])
        insList.append(int(splitSegs[1]))
        capSegs = splitSegs[2].lower().split(' ')
        capList.append(capSegs)

    vd2idx, idx2vd = build_idx_from_list(videoList) 
    igNameList =textread(annIgListFn)
    
    a2d_info_raw= {'cap': capList, 'vd': videoList, 'bbxList': bbxList, 'frmList': frmNameList, 'insList' :  insList, 'igList': igNameList, 'splitDict': splitDict, 'vd2idx': vd2idx, 'idx2vd': idx2vd}
    return a2d_info_raw

    #data = h5py.file() 
def a2dPCK2List(pckFn): 
    a2dDict = pickleload(pckFn)
    imgList = list()
    testCapList = a2dDict['cap']
    for i, cap in enumerate(testCapList):
        vdName = a2dDict['vd'][i]
        frmList = a2dDict['frmList'][i]
        for j, imName in enumerate(frmList):
            imNameFull = vdName+'/'+imName+'.png'
            imgList.append(imNameFull)
    imgList= list(set(imgList)) 
    return imgList

def extAllFrm(annFn, fdPre):
    annDict = pickleload(annFn)
    videoListU = list(set(annDict['vd']))
    frmFullList = list()
    for vdName in videoListU:
        subPre = fdPre + '/' + vdName
        frmNameList = get_specific_file_list_from_fd(subPre, '.png') 
        for frm in frmNameList:
            frmFullList.append(vdName+'/'+frm+'.png')
    return frmFullList

def extAllFrmFn(videoList, fdPre):
    frmvDict = list()
    for vdName in videoList:
        subPre = fdPre + '/' + vdName
        frmNameList = get_specific_file_list_from_fd(subPre, '.png')
        frmNameList.sort()
        frmvDict.append(frmNameList)  
    #pdb.set_trace()
    return frmvDict

def getFrmFn(fdPre, extFn='.jpg'):
    frmListFull = list()
    sub_fd_list = get_list_dir(fdPre) 
    for i, vdName in enumerate(sub_fd_list):
        frmList = get_specific_file_list_from_fd(vdName, extFn, nameOnly=False)
        frmListFull +=frmList
    return frmListFull




if __name__=='__main__':

    fdListFn = '/data1/zfchen/data/actNet/actNetJpgs/'
    #frmList = getFrmFn(fdListFn)
    #pdb.set_trace()

    #pckFn = '../data/annoted_a2d.pd'
    #imPre ='/data1/zfchen/data/A2D/Release/pngs320H'
    #dataAnn = pickleload(pckFn)
    #frmFullList = extAllFrm(pckFn, imPre)
    #pdb.set_trace()
    #dataAnn['frmListGt'] = dataAnn['frmList']
    #dataAnn['frmList'] = frmFullList
    #pickledump('../data/annoted_a2dV2.pd', dataAnn)
    #pdb.set_trace()
    #print('finish')
    #imgList = a2dPCK2List(pckFn)
    #print('finish')
    #annFd = '/disk2/zfchen/data/A2D/Release/sentenceAnno/a2d_annotation_with_instances' 
    #annFn = '/disk2/zfchen/data/A2D/Release/sentenceAnno/a2d_annotation.txt' 
    #annIgListFn = '/disk2/zfchen/data/A2D/Release/sentenceAnno/a2d_missed_videos.txt'
    #annOriFn = '/disk2/zfchen/data/A2D/Release/videoset.csv' 
    #a2dSetParser(annFn, annFd, annIgListFn, annOriFn)
    #otbPKFile ='../data/annForDb_otb.pd'
    #otbPKFileV2 ='../data/annForDb_otbV2.pd'
    #otbNew= fix_otb_frm_bbxList(otbPKFile, otbPKFileV2)
    #imList = otbPCK2List(otbPKFile)
    #print(imList[:10])
    #print(imList[10000])
    #print(len(imList))
