#!/usr/bin/env python
# -*- coding:utf-8 -*-

import commands
import copy
import cPickle
import datetime
import inspect
import json
import os
import pickle
import sys
import time
import scipy.io
import cv2
import pdb

def set_debugger():
    from IPython.core import ultratb
    sys.excepthook = ultratb.FormattedTB(call_pdb=True)

class TimeReporter:
    def __init__(self, max_count, interval=1, moving_average=False):
        self.time           = time.time
        self.start_time     = time.time()
        self.max_count      = max_count
        self.cur_count      = 0
        self.prev_time      = time.time()
        self.interval       = interval
        self.moving_average = moving_average
    def report(self, cur_count=None, max_count=None, overwrite=True, prefix=None, postfix=None, interval=None):
        if cur_count is not None:
            self.cur_count = cur_count
        else:
            self.cur_count += 1
        if max_count is None:
            max_count = self.max_count
        cur_time = self.time()
        elapsed  = cur_time - self.start_time
        if self.cur_count <= 0:
            ave_time = float('inf')
        elif self.moving_average and self.cur_count == 1:
            ave_time = float('inf')
            self.ma_prev_time = cur_time
        elif self.moving_average and self.cur_count == 2:
            self.ma_time      = cur_time - self.ma_prev_time
            ave_time          = self.ma_time
            self.ma_prev_time = cur_time
        elif self.moving_average:
            self.ma_time      = self.ma_time * 0.95 + (cur_time - self.ma_prev_time) * 0.05
            ave_time          = self.ma_time
            self.ma_prev_time = cur_time
        else:
            ave_time = elapsed / self.cur_count
        ETA = (max_count - self.cur_count) * ave_time
        print_str = 'count : %d / %d, elapsed time : %f, ETA : %f' % (self.cur_count, self.max_count, elapsed, ETA)
        if prefix is not None:
            print_str = str(prefix) + ' ' + print_str
        if postfix is not None:
            print_str += ' ' + str(postfix)
        this_interval = self.interval
        if interval is not None:
            this_interval = interval
        if cur_time - self.prev_time < this_interval:
            return
        if overwrite and self.cur_count != self.max_count:
            printr(print_str)
            self.prev_time = cur_time
        else:
            print print_str
            self.prev_time = cur_time

def textread(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '').replace('\r', '')
    return lines

def textdump(path, lines, need_asking=False):
    if os.path.exists(path) and need_asking:
        if 'n' == choosebyinput(['Y', 'n'], path + ' exists. Would you replace? [Y/n]'):
            return False
    f = open(path, 'w')
    for index, i in enumerate(lines):
        try:
            f.write(i.encode("utf-8") + '\n')
        except:
            print(index)
            pdb.set_trace()

    f.close()

def pickleload(path):
    f = open(path)
    this_ans = pickle.load(f)
    f.close()
    return this_ans

def pickledump(path, this_dic):
    f = open(path, 'w')
    this_ans = pickle.dump(this_dic, f)
    f.close()

def cPickleload(path):
    f = open(path, 'rb')
    this_ans = cPickle.load(f)
    f.close()
    return this_ans

def cPickledump(path, this_dic):
    f = open(path, 'wb')
    this_ans = cPickle.dump(this_dic, f, -1)
    f.close()

def jsonload(path):
    f = open(path)
    this_ans = json.load(f)
    f.close()
    return this_ans

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f)
    f.close()

def choosebyinput(cand, message=False):
    if not type(cand) == list and not type(cand) == int:
        print 'The type of cand_list has to be \'list\' or \'int\' .'
        return
    if type(cand) == int:
        cand_list = range(cand)
    if type(cand) == list:
        cand_list = cand
    int_cand_list = []
    for i in cand_list:
        if type(i) == int:
            int_cand_list.append(str(i))
    if message == False:
        message = 'choose by input ['
        for i in int_cand_list:
            message += i + ' / '
        for i in cand_list:
            if not str(i) in int_cand_list:
                message += i + ' / '
        message = message[:-3] + '] : '
    while True:
        your_ans = raw_input(message)
        if your_ans in int_cand_list:
            return int(your_ans)
            break
        if your_ans in cand_list:
            return your_ans
            break

def printr(*targ_str):
    str_to_print = ''
    for temp_str in targ_str:
        str_to_print += str(temp_str) + ' '
    str_to_print = str_to_print[:-1]
    sys.stdout.write(str_to_print + '\r')
    sys.stdout.flush()

def make_red(prt):
    return '\033[91m%s\033[00m' % prt

def emphasize(*targ_str):
    str_to_print = ''
    for temp_str in targ_str:
        str_to_print += str(temp_str) + ' '
    str_to_print = str_to_print[:-1]
    num_repeat = len(str_to_print) / 2 + 1
    print '＿' + '人' * (num_repeat + 1) + '＿'
    print '＞　%s　＜' % make_red(str_to_print)
    print '￣' + 'Y^' * num_repeat + 'Y￣'

def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def makedirs_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def makebsdirs_if_missing(f_path):
    makedirs_if_missing(os.path.dirname(f_path) if '/' in f_path else f_path)

def split_inds(all_num, split_num, split_targ):
    assert split_num >= 1
    assert split_targ >= 0
    assert split_targ < split_num
    part = all_num // split_num
    if not split_num == split_targ+1:
        return split_targ * part, (split_targ+1) * part
    else:
        return split_targ * part, all_num

try:
    import numpy as np
    def are_same_vecs(vec_a, vec_b, this_eps1=1e-5, verbose=False):
        if not vec_a.ravel().shape == vec_b.ravel().shape:
            return False
        if np.linalg.norm(vec_a.ravel()) == 0:
            if not np.linalg.norm(vec_b.ravel()) == 0:
                if verbose:
                    print 'assertion failed.'
                    print 'diff norm : %f' % (np.linalg.norm(vec_a.ravel() - vec_b.ravel()))
                return False
        else:
            if not np.linalg.norm(vec_a.ravel() - vec_b.ravel()) / np.linalg.norm(vec_a.ravel()) < this_eps1:
                if verbose:
                    print 'assertion failed.'
                    print 'diff norm : %f' % (np.linalg.norm(vec_a.ravel() - vec_b.ravel()) / np.linalg.norm(vec_a.ravel()))
                return False
        return True
    def comp_vecs(vec_a, vec_b, this_eps1=1e-5):
        assert are_same_vecs(vec_a, vec_b, this_eps1, True)
    def arrayinfo(np_array):
        print 'max: %04f, min: %04f, abs_min: %04f, norm: %04f,' % (np_array.max(), np_array.min(), np.abs(np_array).min(), np.linalg.norm(np_array)),
        print 'dtype: %s,' % np_array.dtype,
        print 'shape: %s,' % str(np_array.shape),
        print
except:
    def comp_vecs(*input1, **input2):
        print 'comp_vecs() cannot be loaded.'
        return
    def arrayinfo(*input1, **input2):
        print 'arrayinfo() cannot be loaded.'
        return

try:
    import Levenshtein
    def search_nn_str(targ_str, str_lists):
        dist = float('inf')
        dist_str = None
        for i in sorted(str_lists):
            cur_dist = Levenshtein.distance(i, targ_str)
            if dist > cur_dist:
                dist = cur_dist
                dist_str = i
        return dist_str
except:
    def search_nn_str(targ_str, str_lists):
        print 'search_nn_str() cannot be imported.'
        return

def flatten(targ_list):
    new_list = copy.deepcopy(targ_list)
    for i in reversed(range(len(new_list))):
        if isinstance(new_list[i], list) or isinstance(new_list[i], tuple):
            new_list[i:i+1] = flatten(new_list[i])
    return new_list

def predict_charset(targ_str):
    targ_charsets = ['utf-8', 'cp932', 'euc-jp', 'iso-2022-jp']
    for targ_charset in targ_charsets:
        try:
            targ_str.decode(targ_charset)
            return targ_charset
        except UnicodeDecodeError:
            pass
    return None

def remove_non_ascii(targ_str, charset=None):
    if charset is not None:
        assert isinstance(targ_str, str)
        targ_str = targ_str.decode(charset)
    else:
        assert isinstance(targ_str, unicode)
    return ''.join([x for x in targ_str if ord(x) < 128]).encode('ascii')

class StopWatch(object):
    def __init__(self):
        self._time = {}
        self._bef_time = {}
    def tic(self, name):
        self._bef_time[name] = time.time()
    def toc(self, name):
        self._time[name] = time.time() - self._bef_time[name]
        self._time[name] = time.time() - self._bef_time[name]
        return self._time[name]
    def show(self):
        show_str = ''
        for name, elp in self._time.iteritems():
            show_str += '%s: %03.3f, ' % (name, elp)
        printr(show_str[:-2])

Timer = StopWatch # deprecated

def get_free_gpu(default_gpu):
    FORMAT = '--format=csv,noheader'
    COM_GPU_UTIL = 'nvidia-smi --query-gpu=index,uuid ' + FORMAT
    COM_GPU_PROCESS = 'nvidia-smi --query-compute-apps=gpu_uuid ' + FORMAT
    uuid2id = {cur_line.split(',')[1].strip(): int(cur_line.split(',')[0])
               for cur_line in commands.getoutput(COM_GPU_UTIL).split('\n')}
    used_gpus = set()
    for cur_line in commands.getoutput(COM_GPU_PROCESS).split('\n'):
        used_gpus.add(cur_line)
    if len(uuid2id) == len(used_gpus):
        return default_gpu
    elif os.uname()[1] == 'konoshiro':
        return str(1 - int(uuid2id[list(set(uuid2id.keys()) - used_gpus)[0]]))
    else:
        return uuid2id[list(set(uuid2id.keys()) - used_gpus)[0]]

def get_abs_path():
    return os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), __file__)))

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def get_time_str():
    return datetime.datetime.now().strftime('Y%yM%mD%dH%hM%M')

def get_cur_time():
    return str(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

def split_carefully(text, splitter=',', delimiters=['"', "'"]):
    # assertion
    assert isinstance(splitter, str)
    assert not splitter in delimiters
    if not (isinstance(delimiters, list) or isinstance(delimiters, tuple)):
        delimiters = [delimiters]
    for cur_del in delimiters:
        assert len(cur_del) == 1

    cur_ind = 0
    prev_ind = 0
    splitted = []
    is_in_delimiters = False
    cur_del = None
    while cur_ind < len(text):
        if text[cur_ind] in delimiters:
            if text[cur_ind] == cur_del:
                is_in_delimiters = False
                cur_del = None
                cur_ind += 1
                continue
            elif not is_in_delimiters:
                is_in_delimiters = True
                cur_del = text[cur_ind]
                cur_ind += 1
                continue
        if not is_in_delimiters and text[cur_ind] ==  splitter:
            splitted.append(text[prev_ind:cur_ind])
            cur_ind += 1
            prev_ind = cur_ind
            continue
        cur_ind += 1
    splitted.append(text[prev_ind:cur_ind])
    return splitted

def full_listdir(dir_name):
    return [os.path.join(dir_name, i) for i in os.listdir(dir_name)]

def get_list_dir(dir_name):
    fileFdList = full_listdir(dir_name)
    folder_list = list()
    for item in fileFdList:
        if os.path.isdir(item):
            folder_list.append(item)
    return folder_list


class tictoc(object):
    def __init__(self, targ_list):
        self._targ_list = targ_list
        self._list_ind = -1
        self._TR = TimeReporter(len(targ_list))
    def __iter__(self):
        return self
    def next(self):
        self._list_ind += 1
        if self._list_ind > 0:
            self._TR.report()
        if self._list_ind == len(self._targ_list):
            raise StopIteration()
        return self._targ_list[self._list_ind]

ONCE_PRINTED = set()


def print_once(*targ_str):
    frame = inspect.currentframe(1)
    fname = inspect.getfile(frame)
    cur_loc = frame.f_lineno
    cur_key = fname + str(cur_loc)
    if cur_key in ONCE_PRINTED:
        return
    else:
        ONCE_PRINTED.add(cur_key)
    str_to_print = ''
    for temp_str in targ_str:
        str_to_print += str(temp_str) + ' '
    str_to_print = str_to_print[:-1]
    print str_to_print

def get_specific_file_list_from_fd(dir_name, fileType, nameOnly=True):
    list_name = []
    for fileTmp in os.listdir(dir_name):
        file_path = os.path.join(dir_name, fileTmp)
        if os.path.isdir(file_path):
            continue
        elif os.path.splitext(fileTmp)[1] == fileType:
            if nameOnly:
                list_name.append(os.path.splitext(fileTmp)[0])
            else:
                list_name.append(file_path)
    return list_name

# bugs on annotations
def parse_mul_num_lines(fileName, toFloat=True, spliter=','):
    lineOut = []
    lineList= textread(fileName)
    for lineTmp in lineList:
        splitedStr= split_carefully(lineTmp, spliter)
        if(len(splitedStr)<4):
            print('stange encoding for %s!' %(fileName))
            splitedStr= split_carefully(lineTmp, '\t')
            if(len(splitedStr)<4):
                print('stange encoding for %s!' %(fileName))
                splitedStr= split_carefully(lineTmp, ' ')

        if(toFloat):
            splitedTmp= [ float(ele) for ele in splitedStr]
            lineOut.append(splitedTmp)
        else:
            lineOut.append(splitedStr)
    return lineOut

def pck2mat(pckFn, outFn):
    data = pickleload(pckFn)
    scipy.io.savemat(outFn, data)
    print('finish  transformation')

def putCapOnImage(imgVis, capList):
    if isinstance(capList, list):
        cap = ''
        for ele in capList:
            cap +=ele
            cap +=' '
    else:
        cap = capList
    cv2.putText(imgVis, cap, 
            (10, 50),
            cv2.FONT_HERSHEY_PLAIN,
            2, (0, 0, 255),
            2)
    return imgVis

def get_all_file_list(dir_name):
    file_list=list()
    for fileTmp in os.listdir(dir_name):
        file_path = os.path.join(dir_name, fileTmp)
        if os.path.isdir(file_path):
            continue
        file_list.append(file_path)
    return file_list

def resize_image_with_fixed_height(img, hSize =320):
    h, w, c = img.shape
    scl = hSize*1.0/h
    imgResize = cv2.resize(img, None, None,  fx=scl, fy=scl)
    return imgResize, scl,  h, w

