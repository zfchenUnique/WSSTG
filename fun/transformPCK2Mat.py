import os
import sys
sys.path.append('..')
from util.mytoolbox import *
import argparse
from util.base_parser import BaseParser

class trParser(BaseParser):
    def __init__(self, *arg_list, **arg_dict):
        super(trParser, self).__init__(*arg_list, **arg_dict)
        self.add_argument('--pckFile', default='../data/annForDb_otb.pd', type=str)
        self.add_argument('--matFile', default='../data/annForDb_otb.mat', type=str)
        
def parse_args():
    parser = trParser()
    args = parser.parse_args()
    return args

if __name__=='__main__':
    opt = parse_args()

    pck2mat(opt.pckFile, opt.matFile)
