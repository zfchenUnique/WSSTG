import csv
import sys
sys.path.append('..')
from util.mytoolbox import *
import os
import pdb
import subprocess

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

    



if __name__=='__main__':
    vdFd = '/data1/zfchen/data/actNet/actNetUsed/'
    PNGPath = '/data1/zfchen/data/actNet/actNetJpgs'
    TXTPath = '/data1/zfchen/data/actNet/actNetAnn'
    fpsPath = '/data1/zfchen/data/actNet/fpsRate.txt'
    maxH= 320
    makedirs_if_missing(TXTPath)
    annDict = get_all_file_list(vdFd) 
    h5Path = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNetPrpsH5'

    vdh5List = get_specific_file_list_from_fd(h5Path, '.h5')
    fH = open(fpsPath, 'w')
    for i, row in enumerate(annDict):
        #pdb.set_trace()
        vdName = os.path.basename(row).split('.')[0]
        videoPath=row
        pngOutFd = os.path.join( PNGPath, vdName) + '/'
        #if os.path.exists(pngOutFd):
        #    continue
        #pdb.set_trace()
        if vdName+'_rp'  in vdh5List:
            continue

        makedirs_if_missing(pngOutFd)
        #print(videoPath)
        cmdLine = 'ffprobe -v 0 -of csv=p=0 -select_streams 0 -show_entries stream=r_frame_rate %s' %(videoPath)
        #outPut = subprocess.check_output(cmdLine)
        outPut  = os.popen(cmdLine)
        outStr = outPut.read()
        outNum = outStr.replace('\n', '').split('/')
        outString2 = '%s %f\n' %(vdName, float(outNum[0])*1.0/float(outNum[1]))
        fH.write(outString2)
        print(outString2)
        print(outStr)
        #print('finish extracting %s' %(vdName))
        #print('finish resizing %s' %(vdName))
    fH.close()



