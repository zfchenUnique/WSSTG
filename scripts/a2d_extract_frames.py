import csv
import sys
sys.path.append('..')
from util.mytoolbox import *
import os

def parseCSV(csvFn):
    rowList= list()
    with open(csvFn) as f:
        f_csv = csv.reader(f, delimiter=',')
        for row in f_csv:
            rowList.append(row)

    return rowList

if __name__=='__main__':
    a2dAnnFn1 = '/disk2/zfchen/data/A2D/Release/videoset.csv'
    a2dMP4Path = '/disk2/zfchen/data/A2D/Release/clips320H/'
    a2dPNGPath = '/disk2/zfchen/data/A2D/Release/pngs320H/'
    annDict=parseCSV(a2dAnnFn1)
    for i, row in enumerate(annDict):
        vdName = row[0]
        videoPath=a2dMP4Path+vdName+'.mp4'
        pngOutFd = a2dPNGPath + vdName + '/'
        makedirs_if_missing(pngOutFd)
        cmdLine = 'ffmpeg  -i %s %s%s05d.png' %(videoPath, pngOutFd, '%')
        os.system(cmdLine)
        print('finish extracting %s' %(vdName))


