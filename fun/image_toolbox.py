#!/usr/bin/env python
# -*- coding:utf-8 -*-

import commands
import copy
import cv2
from easydict import EasyDict as edict
import numpy as np
import os
import shutil
from dashed_rect import drawrect
#from settings import FFMPEG
import pdb
TMP_DIR = '.tmp'
FFMPEG = 'ffmpeg'

SAVE_VIDEO = FFMPEG + ' -y -r %d -i %s/%s.jpg %s'

def images2video(image_list, frame_rate, video_path, max_edge=None):
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.mkdir(TMP_DIR)
    img_size = None
    for cur_num, cur_img in enumerate(image_list):
        cur_fname = os.path.join(TMP_DIR, '%08d.jpg' % cur_num)
        if max_edge is not None:
            cur_img = imread_if_str(cur_img)
        if isinstance(cur_img, str) or isinstance(cur_img, unicode):
            shutil.copyfile(cur_img, cur_fname)
        elif isinstance(cur_img, np.ndarray):
            max_len = max(cur_img.shape[:2])
            if max_len > max_edge and img_size is None and max_edge is not None:
                magnif = float(max_edge) / float(max_len)
                img_size = (int(cur_img.shape[1] * magnif), int(cur_img.shape[0] * magnif))
                cur_img = cv2.resize(cur_img, img_size)
            elif max_edge is not None:
                if img_size is None:
                    magnif = float(max_edge) / float(max_len)
                    img_size = (int(cur_img.shape[1] * magnif), int(cur_img.shape[0] * magnif))
                cur_img = cv2.resize(cur_img, img_size)
            cv2.imwrite(cur_fname, cur_img)
        else:
            NotImplementedError()
    print commands.getoutput(SAVE_VIDEO % (frame_rate, TMP_DIR, '%08d', video_path))
    shutil.rmtree(TMP_DIR)

def imread_if_str(img):
    if isinstance(img, basestring):
        img = cv2.imread(img)
    return img

def draw_rectangle(img, bbox, color=(0,0,255), thickness=3, use_dashed_line=False):
    img = imread_if_str(img)
    if isinstance(bbox, dict):
        bbox = [
            bbox['x1'],
            bbox['y1'],
            bbox['x2'],
            bbox['y2'],
        ]
    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[0] = min(bbox[0], img.shape[1])
    bbox[1] = min(bbox[1], img.shape[0])
    bbox[2] = max(bbox[2], 0)
    bbox[3] = max(bbox[3], 0)
    bbox[2] = min(bbox[2], img.shape[1])
    bbox[3] = min(bbox[3], img.shape[0])
    assert bbox[2] >= bbox[0]
    assert bbox[3] >= bbox[1]
    assert bbox[0] >= 0
    assert bbox[1] >= 0
    assert bbox[2] <= img.shape[1]
    assert bbox[3] <= img.shape[0]
    cur_img = copy.deepcopy(img)
    #pdb.set_trace()
    if use_dashed_line:
        drawrect(
            cur_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            thickness,
            'dotted'
            )
    else:
        #pdb.set_trace()
        cv2.rectangle(
            cur_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color,
            thickness)
    return cur_img

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def gray_background(img, bbox):
    img = imread_if_str(img)
    if isinstance(bbox, dict):
        bbox = [
            bbox['x1'],
            bbox['y1'],
            bbox['x2'],
            bbox['y2'],
        ]
    bbox[0] = int(max(bbox[0], 0))
    bbox[1] = int(max(bbox[1], 0))
    bbox[0] = min(bbox[0], img.shape[1])
    bbox[1] = min(bbox[1], img.shape[0])
    bbox[2] = int(max(bbox[2], 0))
    bbox[3] = int(max(bbox[3], 0))
    bbox[2] = min(bbox[2], img.shape[1])
    bbox[3] = min(bbox[3], img.shape[0])
    assert bbox[2] >= bbox[0]
    assert bbox[3] >= bbox[1]
    assert bbox[0] >= 0
    assert bbox[1] >= 0
    assert bbox[2] <= img.shape[1]
    assert bbox[3] <= img.shape[0]
    #gray_img = copy.deepcopy(img)
    #gray_img = np.stack((gray_img, gray_img, gray_img), axis=2)
    #gray_img = rgb2gray(gray_img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.stack((gray_img, gray_img, gray_img), axis=2)
    #pdb.set_trace()
    gray_img[bbox[1]:bbox[3], bbox[0]:bbox[2], ...] = img[bbox[1]:bbox[3], bbox[0]:bbox[2], ...]
    #pdb.set_trace()
    return gray_img
