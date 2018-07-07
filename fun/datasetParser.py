import os
import sys
sys.path.append('../')
sys.path.append('../util')
from util.mytoolbox import get_specific_file_list_from_fd, textread, split_carefully, parse_mul_num_lines, pickleload
import pdb

def get_word_list(full_name_list):
    capList =[]
    for filePath in full_name_list: 
        lines = textread(filePath)
        subCapList=[]
        for line in lines:
            wordList=split_carefully(line, ' ')
            subCapList.append(wordList)
        capList.append(subCapList)
    return capList

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

class  otbParser(object):
    def __init__(self, inFd, annoFile, dictFile,  rpFile):
        test_text_fd = inFd + '/OTB_query_test'     
        train_text_fd = inFd + '/OTB_query_train'     
        video_fd = inFd + '/OTB_video'
        
        self.data = pickleload(annoFile)
        self.dict = pickleload(dictFile)
        self.data['qtsFd'] = inFd + '/OTB_query_test'     
        self.data['qtrFd'] = inFd + '/OTB_query_train'     
        self.data['vFd'] = inFd + '/OTB_video'
        
if __name__=='__main__':
    otbPKFile ='../data/annForDb_otb.pd'
    imList = otbPCK2List(otbPKFile)
    print(imList[:10])
    print(imList[10000])
    print(len(imList))
