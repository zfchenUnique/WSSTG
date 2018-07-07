import os
import sys
sys.path.append('../')
sys.path.append('../util')

from gensim.models import KeyedVectors
from  datasetParser import get_otb_data
import numpy as np
from mytoolbox import pickledump, set_debugger
from util.base_parser import BaseParser
import pdb
set_debugger()

class dictParser(BaseParser):
    def __init__(self, *arg_list, **arg_dict):
        super(dictParser, self).__init__(*arg_list, **arg_dict)
        self.add_argument('--dictOutPath', default='../data/dictForDb', type=str)
        self.add_argument('--annoteFd', default='/disk2/zfchen/data/OTB_sentences', type=str)
        self.add_argument('--dictPath', default='/disk2/zfchen/data/dict/GoogleNews-vectors-negative300.bin', type=str)
        self.add_argument('--setName', default='otb', type=str)
        self.add_argument('--setOutPath', default='../data/annForDb', type=str)

def parse_args():
    parser = dictParser()
    args = parser.parse_args()
    return args

def buildVoc(concaList):
    vocList =[]
    for capList in concaList:
        for subCapList in capList:
            for ele in subCapList:
                if ele not in vocList:
                    vocList.append(ele)
    word2idx={}
    idx2word={}
    for i, ele in enumerate(vocList):
        word2idx[ele]=i
        idx2word[i] =ele
    return word2idx, idx2word



if __name__ == '__main__':
    opt = parse_args()
    if opt.setName=='otb':
        
        print('begin parsing dataset: %s\n' %(opt.setName))
        otbInfoRaw= get_otb_data(opt.annoteFd)
        pickledump(opt.setOutPath+'_'+opt.setName+'.pd', otbInfoRaw) 
        print('finish parsing dataset: %s\n' %(opt.setName))
       
        print('begin constructing dictionary for dataset: %s\n' %(opt.setName))
        concaList = otbInfoRaw['trainCap']+otbInfoRaw['testCap']
        word2idx, idx2word= buildVoc(concaList)
        model_word2vec = KeyedVectors.load_word2vec_format(opt.dictPath, binary=True) 
        matrix_word2vec = []
        for word in word2idx.keys():
            try:
                matrix_word2vec.append(model_word2vec[word])
            except:
                KeyError
        matrix_word2vec = np.asarray(matrix_word2vec).astype(np.float32)
        pdb.set_trace()
        outDict = {'idx2word': idx2word, 'word2idx': word2idx, 'word2vec':  matrix_word2vec}
        pickledump(opt.dictOutPath+'_'+opt.setName+'.pd', outDict) 
        print('Finish constructing dictionary for dataset: %s\n' %(opt.setName))
        print('Done!')
