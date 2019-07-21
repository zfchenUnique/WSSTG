#!/usr/bin/env python
# -*- coding:utf-8 -*-

import json
import os
import pdb

EPS = 1e-10


def compute_IoU(bbox1, bbox2):
    bbox1_area = float((bbox1[2] - bbox1[0] + EPS) * (bbox1[3] - bbox1[1] + EPS))
    bbox2_area = float((bbox2[2] - bbox2[0] + EPS) * (bbox2[3] - bbox2[1] + EPS))
    w = max(0.0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + EPS)
    h = max(0.0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + EPS)
    inter = float(w * h)
    ovr = inter / (bbox1_area + bbox2_area - inter)
    return ovr

def is_annotated(traj, frame_ind):
    if not frame_ind in traj:
        return False
    box = traj[frame_ind]
    if box[0] < 0:
        #for el_val in box[1:]:
            #assert el_val < 0
        return False
    #for el_val in box[1:]:
    #    assert el_val >= 0
    return True

def compute_LS(traj, gt_traj):
    # see http://jvgemert.github.io/pub/jain-tubelets-cvpr2014.pdf
    assert isinstance(traj.keys()[0], type(gt_traj.keys()[0]))
    IoU_list = []
    for frame_ind, gt_box in gt_traj.iteritems():
        # make sure the gt_box is within valid

        gt_is_annotated = is_annotated(gt_traj, frame_ind)
        pr_is_annotated = is_annotated(traj, frame_ind)
        if (not gt_is_annotated) and (not pr_is_annotated):
            continue
        if (not gt_is_annotated) or (not pr_is_annotated):
            IoU_list.append(0.0)
            continue
        box = traj[frame_ind]
        IoU_list.append(compute_IoU(box, gt_box))
    return sum(IoU_list) / len(IoU_list)

def jsonload(path):
    f = open(path)
    json_data = json.load(f)
    f.close()
    return json_data

def get_abs_path():
    return os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), __file__)))
