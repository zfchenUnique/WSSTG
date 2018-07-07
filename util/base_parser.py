#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse

class BaseParser(argparse.ArgumentParser):
    def __init__(self, *arg_list, **arg_dict):
        super(BaseParser, self).__init__(*arg_list, **arg_dict)
        self.add_argument('--max_iter', default=1500, type=int)
        self.add_argument('--start_iter', default=0, type=int)
        self.add_argument('--val_interval', default=20, type=int)
        self.add_argument('--saving_interval', default=100, type=int)
        self.add_argument('--suffix', default='')
        self.add_argument('--lrdecay', default=500, type=float)
        self.add_argument('--clip_c', default=10., type=float)
        self.add_argument('--wo_early_stopping',
                          dest='early_stopping',
                          action='store_false',
                          default=True)
        self.add_argument('--alpha', default=0.001, type=float)
        self.add_argument('--lr', default=0.01, type=float)
        self.add_argument('--momentum', default=0.9, type=float)

    def parse_args(self, *arg_list, **arg_dict):
        args = super(BaseParser, self).parse_args(*arg_list, **arg_dict)
        if len(args.suffix) > 0:
            args.suffix = '_' + args.suffix
        return args
