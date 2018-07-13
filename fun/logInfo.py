
import time
import os
class logInF (object):
    def __init__(self, logPre):
        dirname = os.path.dirname(logPre)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        logFile=logPre+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + 'log.txt'
        self.prHandle=open(logFile, 'w')

    def __call__(self, logData):
        print(logData)
        self.prHandle.write(logData)
        self.prHandle.flush()
