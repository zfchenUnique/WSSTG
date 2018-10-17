import csv
import sys
sys.path.append('..')
from util.mytoolbox import *
import os
import pdb
sys.path.append('../annotations')
from ptd_api import *

def resizeImgList(pngOutFd, maxH, annFileName):
    fileList  = get_specific_file_list_from_fd(pngOutFd, '.jpg', nameOnly=False)
    #h, w, scl = [0, 0, 0]
    for i, fileN in enumerate(fileList):
        if i%100==0:
            print('%d /%d %s' %(i, len(fileList), fileN))
        img = cv2.imread(fileN)
        imgRe, scl, h, w = resize_image_with_fixed_height(img, maxH)
        cv2.imwrite(fileN, imgRe)
    with open(annFileName, 'w') as fh:
        fh.write('%d %d %.3f' %(h, w, scl))

def extract_frames_ori():
    vdFd = '/data1/zfchen/data/actNet/actNetUsed/'
    #PNGPath = '/data1/zfchen/data/actNet/actNetJpgs'
    PNGPath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNetJpg'
    TXTPath = '/data1/zfchen/data/actNet/actNetAnn'
    maxH = 320
    makedirs_if_missing(TXTPath)
    annDict = get_all_file_list(vdFd) 

    #vdh5List = get_specific_file_list_from_fd(h5Path, '.h5')
    for i, row in enumerate(annDict):
        #pdb.set_trace()
        vdName = os.path.basename(row).split('.')[0]
        videoPath=row
        pngOutFd = os.path.join( PNGPath, vdName) + '/'
        if os.path.exists(pngOutFd):
            continue
        #pdb.set_trace()
        #if vdName+'_rp'  in vdh5List:
        #    continue

        makedirs_if_missing(pngOutFd)
        cmdLine = 'ffmpeg  -i %s %s%s05d.jpg' %(videoPath, pngOutFd, '%')
        os.system(cmdLine)
        print('finish extracting %s' %(vdName))
        # resize images for saving space
        annFileName = os.path.join(TXTPath +'/', vdName + '.txt')
        resizeImgList(pngOutFd, maxH, annFileName)
        #pdb.set_trace()
        print('finish resizing %s' %(vdName))






if __name__=='__main__':
    
    vdFd = '/data1/zfchen/data/actNet/actNetUsed/'
    #PNGPath = '/data1/zfchen/data/actNet/actNetJpgs'
    PNGPath = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNet/actNetPNG'
    TXTPath = '/data1/zfchen/data/actNet/actNetAnn'
    makedirs_if_missing(TXTPath)
    annDict = get_all_file_list(vdFd) 
    set_list= ['train', 'val', 'test']
    ptd_list = list()
    for set_name in set_list:
        ptd_list.append(PTD(set_name))

    #pdb.set_trace()
    for i, row in enumerate(annDict):
        #pdb.set_trace()
        vdName = os.path.basename(row).split('.')[0]

        fps_tmp=None 
        for ptd in ptd_list:
            vd_name_raw = vdName.split('_', 1)[1]
            if vd_name_raw in ptd.vid2fps.keys():
                fps_tmp = ptd.vid2fps[vd_name_raw]
                break
        assert fps_tmp is not None
        videoPath=row
        pngOutFd = os.path.join( PNGPath, vdName) + '/'
        if os.path.exists(pngOutFd):
            continue
        makedirs_if_missing(pngOutFd)
        cmdLine = 'ffmpeg  -i %s -r %f %s%s05d.png' %(videoPath, fps_tmp, pngOutFd, '%')
        os.system(cmdLine)
        print('finish extracting %s' %(vdName))



