import os
import sys
sys.path.append('../')
sys.path.append('../../annotations')
sys.path.append('../util')
from util.mytoolbox import *
import pdb
import h5py
import csv
import numpy as np
sys.path.append('../annotations')
from script_test_annotation import *
from netUtil import *


def from_pickle_to_h5(vdFd, outFd):
    pdList = get_specific_file_list_from_fd(vdFd, '.pd', nameOnly=False)
    pdList.sort()
    
    vd_name_raw = os.path.basename(vdFd)
    h5_rp_file_name = os.path.join(outFd, vd_name_raw + '_rp.h5')
    h5_ftr_file_name = os.path.join(outFd, vd_name_raw + '_ftr.h5')
    h5_rp_handle = h5py.File(h5_rp_file_name, 'w')
    h5_ftr_handle = h5py.File(h5_ftr_file_name, 'w')
   
    h5_bbx  = h5_rp_handle.create_group('bbx')
    h5_score  = h5_rp_handle.create_group('score')
    h5_imfo = h5_rp_handle.create_group('imfo')

    for i, frmName in enumerate(pdList):
        frmNameRaw = os.path.basename(frmName).split('.')[0]
        #print('%s %d' %(frmName, i))
        pk_handle =  pickleload(frmName)
        h5_bbx[frmNameRaw] = pk_handle['rois']
        h5_score[frmNameRaw] = pk_handle['roisS']
        h5_imfo[frmNameRaw] = pk_handle['imFo']

        h5_ftr_handle[frmNameRaw] = pk_handle['roiFtr']

    h5_rp_handle.close()
    h5_ftr_handle.close()
    print('finish transforming %s\n' %(vdFd))

def check_h5_pd(vdFd, pk_ftr_Fn, pk_rp_Fn):
    f_ftr = h5py.File(pk_ftr_Fn, 'r')
    f_rp = h5py.File(pk_rp_Fn, 'r')
    
    pdList = get_specific_file_list_from_fd(vdFd, '.pd', nameOnly=False)
    for i, frmName in enumerate(pdList):
        frmNameRaw = os.path.basename(frmName).split('.')[0]
        #print('%s %d' %(frmName, i))
        pk_handle =  pickleload(frmName)
        maxV = np.max(f_ftr[frmNameRaw][()] - pk_handle['roiFtr'])
        maxV2 = np.max(f_rp['bbx'][frmNameRaw][()] - pk_handle['rois'])
        maxV3 = np.max(f_rp['score'][frmNameRaw][()] - pk_handle['roisS'])
        maxV4 = np.max(f_rp['imfo'][frmNameRaw][()] - pk_handle['imFo'])
        print(maxV + maxV2 + maxV3 + maxV4)
        #pdb.set_trace()
    f.close()

def transform_all_vd_from_pk_to_h5(rpFd, outFd):
    vdListFull = get_list_dir(rpFd)
    for vdName in vdListFull:
        print(vdName)

def extract_shot_prp_list(shot, rpPath, tube_prp_num, do_norm=True):
    frmList = get_shot_frames(shot)
     
    vdName = shot.video_id
    vd_prp_file_path = os.path.join(rpPath, 'v_' + vdName + '_rp.h5')
    f_rp = h5py.File(vd_prp_file_path, 'r')
    keyList = f_rp['bbx'].keys()
    prp_list = list()
    print(len(keyList))
    #pdb.set_trace()
    for i, frmName in enumerate(frmList):
        print('%d\%d, %s\n' %(i, len(keyList), frmName))
        tmp_bbx = f_rp['bbx'][frmName][()][:tube_prp_num]
        tmp_score = f_rp['score'][frmName][()][:tube_prp_num]
        tmp_info = f_rp['imfo'][frmName][()].squeeze()
        if do_norm:
            tmp_bbx[:, 0] = tmp_bbx[:, 0]/tmp_info[1]
            tmp_bbx[:, 2] = tmp_bbx[:, 2]/tmp_info[1]
            tmp_bbx[:, 1] = tmp_bbx[:, 1]/tmp_info[0]
            tmp_bbx[:, 3] = tmp_bbx[:, 3]/tmp_info[0]
        else:
            tmp_bbx = tmp_bbx/tmp_info[2]
        tmp_score = np.expand_dims(tmp_score, axis=1 )
        prp_list.append([tmp_score, tmp_bbx])
    pdb.set_trace()
    return prp_list, frmList

def extract_shot_prp_list_ori(shot, rpPath, tube_prp_num, do_norm=True):
    frmList = get_shot_frames(shot)
     
    vdName = shot.video_id
    vd_prp_file_folder = os.path.join(rpPath, 'v_' + vdName)
    prp_list = list()
    #pdb.set_trace()
    for i, frmName in enumerate(frmList):
        print('%d\%d, %s\n' %(i, len(frmList), frmName))
        vd_prp_file_folder_frm_fn = os.path.join(vd_prp_file_folder, frmName+'.pd')
        f_handle = pickleload(vd_prp_file_folder_frm_fn)
        tmp_bbx = f_handle['rois'][:tube_prp_num]
        tmp_score = f_handle['roisS'][:tube_prp_num]
        tmp_info = f_handle['imFo'].squeeze()
        if do_norm:
            tmp_bbx[:, 0] = tmp_bbx[:, 0]/tmp_info[1]
            tmp_bbx[:, 2] = tmp_bbx[:, 2]/tmp_info[1]
            tmp_bbx[:, 1] = tmp_bbx[:, 1]/tmp_info[0]
            tmp_bbx[:, 3] = tmp_bbx[:, 3]/tmp_info[0]
        else:
            tmp_bbx = tmp_bbx/tmp_info[2]
        tmp_score = np.expand_dims(tmp_score, axis=1 )
        prp_list.append([tmp_score, tmp_bbx])
    #pdb.set_trace()
    return prp_list, frmList


def get_shot_tube_proposals(shot, rpPath, tube_prp_num, tube_thre=0.2, tube_out_path=None):
    prp_list, frm_list = extract_shot_prp_list(shot, rpPath, tube_prp_num)
    results = get_tubes(prp_list, tube_thre)
    return results, frm_list

def get_shot_tube_proposals_ori(shot, rpPath, tube_prp_num, tube_thre=0.2, tube_out_path=None):
    prp_list, frm_list = extract_shot_prp_list_ori(shot, rpPath, tube_prp_num)
    results = get_tubes(prp_list, tube_thre)
    return results, frm_list



def resize_tube_bbx(tube_vis, frmImList_vis):
    for prpId, frm in enumerate(tube_vis):
        h, w, c = frmImList_vis[prpId].shape
        tube_vis[prpId][0] = tube_vis[prpId][0]*w
        tube_vis[prpId][2] = tube_vis[prpId][2]*w
        tube_vis[prpId][1] = tube_vis[prpId][1]*h
        tube_vis[prpId][3] = tube_vis[prpId][3]*h
    return tube_vis

def evaluate_set_tube_recall(rpPath, set_name = 'test', connect_thre = 0.2, topK=20):
    ptd = PTD(set_name)
    shotList = ptd.shots
    recall_05 = [0.0] * (topK + 1)
    recall_02 = [0.0] * (topK + 1)
    personNum = 0
    for shotId, shotInfo in enumerate(shotList):
        shot = ptd.shot(shotId+1)
        #if(shotId<39):
        #    continue
        print('vid: %d, person Num: %d' %(shotId+1, personNum))
        prp_list, frm_list = extract_shot_prp_list(shot, rpPath, topK, do_norm=True)
        results = get_tubes(prp_list, connect_thre)
        shot_proposals = [results, frm_list]
        for ii, person_in_shot in enumerate(shot.people):
            recall_k_5 = evaluate_tube_recall(shot_proposals, shot, person_in_shot, thre=0.5 ,topKOri=20)
            recall_k_2 = evaluate_tube_recall(shot_proposals, shot, person_in_shot, thre=0.2 ,topKOri=20)
            personNum +=1
            for i in range(topK+1):
                recall_02[i] +=recall_k_2[i]
                recall_05[i] +=recall_k_5[i]
            print(np.array(recall_02)/personNum)
            print(np.array(recall_05)/personNum)

def extract_set_tubes(rpPath, connect_thre = 0.2, set_name='test', topK=20, outPath='./'):
    ptd = PTD(set_name)
    shotList = ptd.shots
    personNum = 0
    outSubFd = os.path.join(outPath, set_name)
    makedirs_if_missing(outSubFd)
    for shotId, shotInfo in enumerate(shotList):
       
        #pdb.set_trace()
        outName = outSubFd+'/' +str(shotId+1) + '.pk'
        shot = ptd.shot(shotId+1)
        #print('vid Name: %s, vid: %d, person Num: %d' %(shot.video_id, shotId+1, personNum))
        if os.path.isfile(outName):
            personNum +=1
            continue
        shot = ptd.shot(shotId+1)
        print('vid Name: %s, vid: %d, person Num: %d' %(shot.video_id, shotId+1, personNum))
        try:
            prp_list, frm_list = extract_shot_prp_list(shot, rpPath, topK, do_norm=True)
        except:
            print('%s do not  exist:' %(shot.video_id))
            personNum +=1
            continue
        results = get_tubes(prp_list, connect_thre)
        shot_proposals = [results, frm_list]
        pickledump(outSubFd+'/' +str(shotId+1) + '.pk', shot_proposals)
        personNum +=1

def extract_set_tubes_ori(rpPath, connect_thre = 0.2, set_name='test', topK=20, outPath='./'):
    ptd = PTD(set_name)
    shotList = ptd.shots
    personNum = 0
    outSubFd = os.path.join(outPath, set_name)
    makedirs_if_missing(outSubFd)
    for shotId, shotInfo in enumerate(shotList):
       
        #pdb.set_trace()
        outName = outSubFd+'/' +str(shotId+1) + '.pk'
        shot = ptd.shot(shotId+1)
        print('vid Name: %s, vid: %d, person Num: %d' %(shot.video_id, shotId+1, personNum))
        if os.path.isfile(outName):
            personNum +=1
            continue
        shot = ptd.shot(shotId+1)
        print('vid Name: %s, vid: %d, person Num: %d' %(shot.video_id, shotId+1, personNum))
        try:
            prp_list, frm_list = extract_shot_prp_list_ori(shot, rpPath, topK, do_norm=True)
        except:
            print('%s do not  exist:' %(shot.video_id))
            personNum +=1
            continue
        results = get_tubes(prp_list, connect_thre)
        shot_proposals = [results, frm_list]
        pickledump(outSubFd+'/' +str(shotId+1) + '.pk', shot_proposals)
        personNum +=1


def capiton_to_word_list(des_str):
    import string
    for c in string.punctuation:
        des_str = des_str.replace(c, '')
    return split_carefully(des_str.lower().replace('.', '').replace(',', '').replace("\'", '').replace('-', '').replace('\n', '').replace('\r', '').replace('\"', '').rstrip().replace("\\",'').replace('?', '').replace('/','').replace('#','').replace('(', '').replace(')','').replace(';','').replace('!', '').replace('/',''), ' ')

def build_actNet_word_list():
    set_list = ['train', 'val', 'test']
    des_list = list()
    for set_name in set_list:
        ptd = PTD(set_name)
        descriptions_list = ptd.descriptions
        for disId, disInfo in enumerate(descriptions_list):
            des_str = ptd.description(disId+1).description
            wordList = capiton_to_word_list(des_str)
            #pdb.set_trace()
            des_list +=wordList
    des_list = list(set(des_list))
    return des_list

def get_actNet_image_list():
    actNet_png_path = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNet/actNetPNG'
    fn_fd_list = os.listdir(actNet_png_path)
    ptd_list = {}
    frm_list = list()
    #pdb.set_trace()
    #fn_fd_list  = fn_fd_list[:10]

    for i, folder_name in enumerate(fn_fd_list):
        print('%d/ %d %s\n' %(i, len(fn_fd_list), folder_name))
        tmp_full_dir_path = os.path.join(actNet_png_path, folder_name)
        if os.path.isdir(tmp_full_dir_path):
            vd_name = folder_name.split('_', 1)[1]
            shotInfo = video2shot(vd_name, ptd_list)
            for ii, shot in enumerate(shotInfo):
                set_name, shot_id = shot
                if set_name in ptd_list.keys():
                    ptd_tmp = ptd_list[set_name]
                else:
                    ptd_tmp = PTD(set_name)
                    ptd_list[set_name] = ptd_tmp
                tmp_shot = ptd_tmp.shot(shot_id)
                ann_frm_st = min(tmp_shot.first_frame-1, 0)
                ann_frm_ls = tmp_shot.last_frame
                for iii, frm_id in enumerate(range(ann_frm_st, ann_frm_ls)):
                    frm_name = '%05d.png' %(frm_id+1)
                    img_full_path = os.path.join(actNet_png_path, 'v_'+ vd_name, frm_name)
                    frm_list.append(img_full_path)
        
    frm_list = list(set(frm_list))
    frm_list.sort()
    return frm_list

def vis_video_prp_actNet():
    rpPath = '/data1/zfchen/data/remote_disk/data10/actNet'
    tubeRpPath = '/data1/zfchen/data/remote_disk/data10/actNet_tube_prp'
    imgFolder = '/mnt/ceph_cv/aicv_image_data/forestlma/zfchen/actNet/actNetPNG'
    #vdName = '8rimo9x4qqw'
    #vdName = '--veKG73Di4'
    vdName = '-4VuHlphgL4'
    #vdName = 'jl10JmELMqY'
    #vdName = 'F99Suh6SvD8'
    outFd = './sample'
    tube_prp_num = 30
    vis_frame_num = 30
    topK =30
    pdb.set_trace()
    shotList = video2shot(vdName) 
    for i, shotInfo in enumerate(shotList):
        set_name, shotId = shotInfo
        ptd = PTD(set_name)
        shot=ptd.shot(shotId)
        for i, des_obj in enumerate(shot.descriptions):
            print('%s\n' %(des_obj.description))

        prp_list, frm_list = extract_shot_prp_list_ori(shot, rpPath, tube_prp_num, do_norm=True)
        pdb.set_trace()
        results = get_tubes(prp_list, 0.2)
        # test recall tubes
        shot_proposals = [results, frm_list]
        for ii, person_in_shot in enumerate(shot.people):
            recall_k = evaluate_tube_recall(shot_proposals, shot, person_in_shot, thre=0.5 ,topKOri=topK)
            print(recall_k)
            recall_k = evaluate_tube_recall(shot_proposals, shot, person_in_shot, thre=0.2 ,topKOri=topK)
            print(recall_k)

        #continue
        # visualize tubes
        frmImNameList = get_shot_frames_full_path(shot, imgFolder, '.png')
        pdb.set_trace() 
        frmImList = list()
        for fId, imPath  in enumerate(frmImNameList):
            img = cv2.imread(imPath)
            frmImList.append(img)

        visIner = int(len(frmImList) /vis_frame_num )
        for ii in range(min(len(results[0]), 5)):
            print('visualizing tube %d\n'%(ii))
            tube = results[0][ii]
            frmImList_vis = [frmImList[iii] for iii in range(0, len(frmImList), visIner)]
            tube_vis = [tube[iii] for iii in range(0, len(frmImList), visIner)]
            tube_vis_resize = resize_tube_bbx(tube_vis, frmImList_vis)
            visTube_from_image(copy.deepcopy(frmImList_vis), tube_vis_resize, 'sample/'+vdName + str(ii)+'.gif')

if __name__ == '__main__':
    vis_video_prp_actNet() 
    #from_img_prp_to_tube(vdName, rpPath, tubeRpPath,  tube_prp_num)

    #rpFd = '/data1/zfchen/data/actNet/actNetPrps/v_19YCgLDhfoE'
    #outFd = '/data1/zfchen/data/actNet/actNetPrpsH5'
    #makedirs_if_missing(outFd)
    #from_pickle_to_h5(rpFd, outFd)
    #pk_rp_Fn = os.path.join(outFd, os.path.basename(rpFd)+'_rp.h5')
    #pk_ftr_Fn = os.path.join(outFd, os.path.basename(rpFd)+'_ftr.h5')

    #check_h5_pd(rpFd, pk_ftr_Fn, pk_rp_Fn)
