import os
import sys
sys.path.append('../')
sys.path.append('../util')

from gensim.models import KeyedVectors
from  datasetParser import *
import numpy as np
from mytoolbox import pickledump, set_debugger
from util.base_parser import BaseParser
import pdb
from actNetDatasetParser import *
from vidDatasetParser import *

set_debugger()

class dictParser(BaseParser):
    def __init__(self, *arg_list, **arg_dict):
        super(dictParser, self).__init__(*arg_list, **arg_dict)
        self.add_argument('--dictOutPath', default='../data/dictForDb', type=str)
        self.add_argument('--annoteFd', default='/disk2/zfchen/data/OTB_sentences', type=str)
        self.add_argument('--dictPath', default='/disk2/zfchen/data/dict/GoogleNews-vectors-negative300.bin', type=str)
        self.add_argument('--setName', default='otb', type=str)
        self.add_argument('--setOutPath', default='../data/annForDb', type=str)
        self.add_argument('--annFn', default='', type=str)
        self.add_argument('--annFd', default='', type=str)
        self.add_argument('--annIgListFn', default='', type=str)
        self.add_argument('--annOriFn', default='', type=str)
            
            

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

def buildVocA2d(concaList):
    vocList =[]
    for capList in concaList:
        for ele in capList:
            if ele not in vocList:
                vocList.append(ele)
    word2idx={}
    idx2word={}
    for i, ele in enumerate(vocList):
        word2idx[ele]=i
        idx2word[i] =ele
    return word2idx, idx2word

def buildVocActNet(vocListOri):
    vocList = list()
    for ele in vocListOri:
        if ele not in vocList:
            vocList.append(ele)
    word2idx={}
    idx2word={}
    for i, ele in enumerate(vocList):
        word2idx[ele]=i
        idx2word[i] =ele
    return word2idx, idx2word

def build_word_vec(word_list, model_word2vec):
    matrix_word2vec = []
    igNoreList =  list()
    for i, word in enumerate(word_list):
        print(i, word)
        try:
            matrix_word2vec.append(model_word2vec[word])
        except:
            igNoreList.append(word)
            #matrix_word2vec.append(np.zeros((300), dtype=np.float32))
            randArray=np.random.rand((300)).astype('float32')
            matrix_word2vec.append(randArray)
            try:
                print('%s is not the vocaburary'% word)
            except:
                print('fail to print the word!')
            pdb.set_trace()
    return matrix_word2vec, igNoreList

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
        pdb.set_trace()
        igNoreList =  list()
        for i, word in enumerate(word2idx.keys()):
            print(i, word)
            try:
                matrix_word2vec.append(model_word2vec[word])
            except:
                pdb.set_trace()
                igNoreList.append(word)
                #matrix_word2vec.append(np.zeros((300), dtype=np.float32))
                matrix_word2vec.append(np.random.rand((300), dtype=np.float32))
                print('%s is not the vocaburary'% word)
        matrix_word2vec = np.asarray(matrix_word2vec).astype(np.float32)
        pdb.set_trace()
        outDict = {'idx2word': idx2word, 'word2idx': word2idx, 'word2vec':  matrix_word2vec, 'out_voca': igNoreList}
        pickledump(opt.dictOutPath+'_'+opt.setName+'.pd', outDict) 
        print('Finish constructing dictionary for dataset: %s\n' %(opt.setName))
        print('Done!')

    elif opt.setName=='a2d':
        print('begin parsing dataset: %s\n' %(opt.setName))
        outAnnName = opt.setOutPath+'_'+opt.setName+'.pd'
        if not os.path.isfile(outAnnName):
            a2dInfoRaw = a2dSetParser(opt.annFn, opt.annFd, opt.annIgListFn, opt.annOriFn)
            pickledump(outAnnName, a2dInfoRaw)
        else:
            a2dInfoRaw = pickleload(outAnnName) 
        print('finish parsing dataset: %s\n' %(opt.setName))
       
        print('begin constructing dictionary for dataset: %s\n' %(opt.setName))
        word2idx, idx2word= buildVocA2d(a2dInfoRaw['cap'])
        model_word2vec = KeyedVectors.load_word2vec_format(opt.dictPath, binary=True) 
        matrix_word2vec = []
        igNoreList =  list()
        for i, word in enumerate(word2idx.keys()):
            print(i, word)
            try:
                matrix_word2vec.append(model_word2vec[word])
            except:
                pdb.set_trace()
                igNoreList.append(word)
                #matrix_word2vec.append(np.zeros((300), dtype=np.float32))
                randArray=np.random.rand((300)).astype('float32')
                matrix_word2vec.append(randArray)
                print('%s is not the vocaburary'% word)
        matrix_word2vec = np.asarray(matrix_word2vec).astype(np.float32)
        pdb.set_trace()
        outDict = {'idx2word': idx2word, 'word2idx': word2idx, 'word2vec':  matrix_word2vec, 'out_voca': igNoreList}
        pickledump(opt.dictOutPath+'_'+opt.setName+'.pd', outDict) 
        print('Finish constructing dictionary for dataset: %s\n' %(opt.setName))
        
    elif opt.setName=='actNet':
        print('begin parsing dataset: %s\n' %(opt.setName))
        word_list = build_actNet_word_list()
        print(len(word_list))
        pdb.set_trace()
        word2idx, idx2word= buildVocActNet(word_list)
        model_word2vec = KeyedVectors.load_word2vec_format(opt.dictPath, binary=True) 
        matrix_word2vec, igNoreList = build_word_vec(word2idx.keys(), model_word2vec)
        matrix_word2vec = np.asarray(matrix_word2vec).astype(np.float32)
        pdb.set_trace()

        outDict = {'idx2word': idx2word, 'word2idx': word2idx, 'word2vec':  matrix_word2vec, 'out_voca': igNoreList}
        pickledump(opt.dictOutPath+'_'+opt.setName+'.pd', outDict) 
        print('Finish constructing dictionary for dataset: %s\n' %(opt.setName))

    elif opt.setName=='vid':
        print('begin parsing dataset: %s\n' %(opt.setName))
        word_list = build_vid_word_list()
        print(len(word_list))
        word2idx, idx2word= buildVocActNet(word_list)
        model_word2vec = KeyedVectors.load_word2vec_format(opt.dictPath, binary=True) 
        matrix_word2vec, igNoreList = build_word_vec(word2idx.keys(), model_word2vec)
        matrix_word2vec = np.asarray(matrix_word2vec).astype(np.float32)
        pdb.set_trace()

        outDict = {'idx2word': idx2word, 'word2idx': word2idx, 'word2vec':  matrix_word2vec, 'out_voca': igNoreList}
        pickledump(opt.dictOutPath+'_'+opt.setName+'_v2.pd', outDict) 
        print('Finish constructing dictionary for dataset: %s\n' %(opt.setName))






