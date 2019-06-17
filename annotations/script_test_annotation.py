from ptd_api import *
import cv2
import copy
import numpy as np
import shutil
import commands
import sys
sys.path.append('../')
from util.mytoolbox import *

TMP_DIR = '.tmp'
FFMPEG = 'ffmpeg'
SAVE_VIDEO = FFMPEG + ' -y -r %d -i %s/%s.jpg %s'

def draw_rectangle(img, bbox, color=(0,0,255), thickness=3):
    img = imread_if_str(img)
    if isinstance(bbox, dict):
        bbox = [
            bbox['x1'],
            bbox['y1'],
            bbox['x2'],
            bbox['y2'],
        ]
    assert bbox[2] >= bbox[0]
    assert bbox[3] >= bbox[1]
    assert bbox[0] >= 0
    assert bbox[1] >= 0
    assert bbox[2] <= img.shape[1]
    assert bbox[3] <= img.shape[0]
    cur_img = copy.deepcopy(img)
    cv2.rectangle(
        cur_img,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        color,
        thickness)
    return cur_img

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

def video2shot(vdName, ptd_list = {}):
    setNameList = ['train', 'val', 'test']
    shotInfo = list()
    for set_name in setNameList:
        if set_name in ptd_list.keys():
            ptd = ptd_list[set_name]
        else:
            ptd = PTD(set_name)
        shotLgh = len(ptd.id2shot)
        for i in range(shotLgh):
            vdNameShot = ptd.shot(i+1).video_id
            if(vdName==vdNameShot):
                shotInfo.append([set_name, i+1])
    return shotInfo

def imread_if_str(img):
    if isinstance(img, basestring):
        img = cv2.imread(img)
    return img

def get_shot_frames(shot):
    annFrmSt = shot.first_frame
    annFrmLs = shot.last_frame
    tmpFrmList = list(range(annFrmSt, annFrmLs+1))
    frmList = list()
    for i, frmIdx in enumerate(tmpFrmList):
        strIdx = '%05d' %(frmIdx)
        frmList.append(strIdx)
    return frmList

def get_shot_frames_full_path(shot, preFd, fn_ext='.jpg'):
    vd_name = shot.video_id 
    vd_subPath = os.path.join(preFd, 'v_' + vd_name)
    framList = get_shot_frames(shot)
    framListFull = list()
    for frm in framList:
        tmpFrm = os.path.join(vd_subPath, frm + fn_ext)
        framListFull.append(tmpFrm)
    return framListFull

# shot_proposals : [tubes, frameList]
# 
def evaluate_tube_recall(shot_proposals, shot, person_in_shot, thre=0.5 ,topKOri=20):
#    pdb.set_trace()
    topK = min(topKOri, len(shot_proposals[0][0]))
    recall_k = [0.0] * (topK + 1)
    boxes = {}
    for frame_ind, box in zip(shot.annotated_frames, person_in_shot['boxes']):
        keyName = '%05d' %(frame_ind)
        boxes[keyName] = box

    #pdb.set_trace()
    tube_list, frame_list = shot_proposals
    assert(len(tube_list[0][0])== len(frame_list))
    is_person_annotated = False
    for i in range(topK):
        recall_k[i+1] = recall_k[i]
        if is_person_annotated:
            continue
        curTubeOri = tube_list[0][i]
        tube_key_bbxList = {}
        for frame_ind, gt_box in boxes.iteritems():
            try:
                index_tmp = frame_list.index(frame_ind)
                tube_key_bbxList[frame_ind] = curTubeOri[index_tmp]
            except:
                print('key %s do not exist in shot' %(frame_ind))
        #pdb.set_trace()
        ol = compute_LS(tube_key_bbxList, boxes) 
        if ol < thre:
            continue
        else:
            recall_k[i+1] += 1.0
            is_person_annotated = True
    return recall_k

def vis_ann_tube():
    pngFolder ='/data1/zfchen/data/actNet/actNetJpgs'
    annFolder ='/data1/zfchen/data/actNet/actNetAnn'
    #vdName = '2Peh_gdQCjg' 
    #vdName = 'jqKK2KH6l4Q' 
    #vdName = 'W4tmb8RwzQM' 
    vdName = 'ydRycaBjMVw' 
    shotList = video2shot(vdName)

    pdb.set_trace()

    txtFn = os.path.join(annFolder, 'v_'+vdName + '.txt')
    fH = open(txtFn)
    vdInfoStr = fH.readlines()[0]
    vdInfo = [float(ele) for ele in vdInfoStr.split(' ')]
    insNum = 1

    for i, shotInfo in enumerate(shotList):
        set_name, shotId  = shotInfo
        ptd = PTD(set_name)
        shot  = ptd.shot(shotId)        
        annFtrList = shot.annotated_frames
        for ii, person in enumerate(shot.people):
            imgList = list()
            print(person.descriptions[0].description)
            for iii in range(len(annFtrList)):
                frmName = '%05d' %(annFtrList[iii]+1)
                imFn = pngFolder + '/v_' + shot.video_id + '/' + frmName + '.jpg'
                img = cv2.imread(imFn)
                h, w, c = img.shape
                bbox = person.boxes[iii]
                #pdb.set_trace()
                bbox[0] = bbox[0]*w
                bbox[2] = bbox[2]*w
                bbox[1] = bbox[1]*h
                bbox[3] = bbox[3]*h
                #pdb.set_trace()
                imgNew = draw_rectangle(img, bbox, color=(0,0,255), thickness=3)
                imgList.append(imgNew)
            images2video(imgList, 10, './' + str(insNum)+'.gif')
            insNum +=1

def extract_instance_caption_list():
    set_list = ['train', 'test', 'val']
    outPre = './data/cap_list_'
    for i, set_name in enumerate(set_list):
        ptd = PTD(set_name)
        outFn = outPre + set_name + '.txt'
        desption_list_dict = ptd.descriptions
        description_list = [ des_dict['description'] for des_dict in desption_list_dict]
        textdump(outFn, description_list)

if __name__=='__main__':
   extract_instance_caption_list() 
