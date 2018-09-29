import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def parse_args():

    parse = argparse.ArgumentParser(description='parse loss curve from log File')
    parse.add_argument('--filePath', default='../data/log/2017-07-24 17:45:19log.txt', type =str)
    parse.add_argument('--savePath', default='loss.png', type =str)
    parse.add_argument('--meanNum', default=10, type=int)

    return parse.parse_args()

if __name__=='__main__':
    param = parse_args()
    fHand = open(param.filePath)
    savePath = param.savePath
    meanNum = param.meanNum
    data = fHand.readlines()
    data = [x.strip() for x in data]
    trainingLossFlag =[]
    iterationFlag    =[]
    testLossFlag     =[]
    testIterFlag     =[]

    baseIter =0
    iterEpoch =0
    for iterIdx, dataLine in enumerate(data):
        if(iterIdx%meanNum!=0):
            continue
        trainFlag=dataLine.find('Training Cls loss')
        if(trainFlag!=-1):
            dataList = dataLine.split(', ')
            trainingLossFlag.append(float(dataList[-2]))
            iterStr = (dataList[1])
            iterVec=iterStr.split(' ')
            iter =float(iterVec[-1])
            epoch =float(iterVec[1])
            if(len(iterationFlag)>0 and iter<iterationFlag[-1] and iterEpoch==0):
                iterEpoch =iterationFlag[-1]
            iterationFlag.append(epoch*iterEpoch+int(iter))
        testFlag =dataLine.find('Validation loss')
        if(testFlag!=-1):
            try:
                dataList = dataLine.split(', ')
                testLossFlag.append(float(dataList[2]))
                iterStr = dataList[1]
                iterVec=iterStr.split(' ')
                iter =float(iterVec[-1])
                epoch =float(iterVec[1])
                testIterFlag.append(epoch*iterEpoch+int(iter))
            except:
                print 'log file of Bad fomat'

    fig, ax1 = plt.subplots()
    ax1.plot(iterationFlag, trainingLossFlag, testIterFlag, testLossFlag)
    #plt.axis([0, 6000, 0, 0.8])
    #plt.show()
    #plt.ylim((0, 0.8))
    fig.savefig(savePath)
    plt.close()
    




