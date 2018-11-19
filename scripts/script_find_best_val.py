import os
from tensorboardX import SummaryWriter
import pdb
import tensorflow as tf

def get_specific_file_list_from_fd(dir_name, fileType, nameOnly=True):
    list_name = []
    for fileTmp in os.listdir(dir_name):
        file_path = os.path.join(dir_name, fileTmp)
        if os.path.isdir(file_path):
            continue
        elif os.path.splitext(fileTmp)[-1] == fileType:
            if nameOnly:
                list_name.append(os.path.splitext(fileTmp)[0])
            else:
                list_name.append(file_path)
    return list_name

def find_best_epoch(file_name, epNum=30, tag_name='Average_testing_accuracy', tag_check='Average_validation_accuracy', ep_itr=6582):
    #pdb.set_trace()
    best_ele = None
    epId = 0
    for e in tf.train.summary_iterator(file_name):
        for v in e.summary.value:
            if v.tag == tag_name:
                if best_ele is None or v.simple_value > best_ele.summary.value[0].simple_value:
                    best_ele = e
                epId +=1
        if epId>=epNum:
            break


    ele_check = None
    for e in tf.train.summary_iterator(file_name):
        if e.step!=best_ele.step:
            continue
        for v in e.summary.value:
            if v.tag == tag_check:
                ele_check = e
                break
    print(best_ele)
    print(ele_check)
    print('finding in %d epoch\n'%(epId))
    print('epoch id %f\n' %(best_ele.step/ep_itr))
    print('finish!\n')

def find_best_epoch_topK(file_name, epNum=30, tag_name='Average_testing_accuracy', tag_check='Average_validation_accuracy', ep_itr=6582, topK=1):
    #pdb.set_trace()
    best_ele_list = list()
    best_value_list = list()
    minValue = 0
    minIndex = -1
    epId = 0
    #pdb.set_trace()
    for e in tf.train.summary_iterator(file_name):
        for v in e.summary.value:
            if v.tag == tag_name:
                if len(best_ele_list)< topK:
                    best_ele_list.append(e)
                    sorted_idx_list  = sorted(range(len(best_ele_list)), key=lambda k: best_ele_list[k].summary.value[0].simple_value)
                    minValue = best_ele_list[sorted_idx_list[0]].summary.value[0].simple_value
                    minIndex = sorted_idx_list[0]
                    continue
                if  v.simple_value > minValue:
                    best_ele_list[minIndex] = e
                    sorted_idx_list  = sorted(range(len(best_ele_list)), key=lambda k: best_ele_list[k].summary.value[0].simple_value)
                    minValue = best_ele_list[sorted_idx_list[0]].summary.value[0].simple_value
                    minIndex = sorted_idx_list[0]

                epId +=1
        if epId>=epNum:
            break

    ele_check_list = list()
    step_list = [ele.step for ele in best_ele_list]
    for e in tf.train.summary_iterator(file_name):
        if e.step not in step_list:
            continue
        for v in e.summary.value:
            if v.tag == tag_check:
                ele_check_list.append(e)
                break
    print(best_ele_list)
    print(ele_check_list)
    print('######################################################################\n')
    
    # Finding best iterations in checked epoch
    sorted_idx_list  = sorted(range(len(ele_check_list)), key=lambda k: ele_check_list[k].summary.value[0].simple_value)
    best_idx = sorted_idx_list[-1]
    best_check_ele = ele_check_list[best_idx]
    val_ele = None
    for ele_val in best_ele_list:
        if ele_val.step==best_check_ele.step:
            val_ele = ele_val
    print(val_ele)
    print(best_check_ele)
    print('finding in %d epoch\n'%(epId))
    print('epoch id %f\n' %(best_check_ele.step/ep_itr))
    print('finish!\n')

if __name__=='__main__':
    # avg + dvsa
    # path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_fc_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.02018-10-24 14:38:49'
    # vlad + dvsa
    # path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_vlad_v1_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.0_centre_32_hidden_dim_1282018-10-24 14:38:22'
    # lstm + dvsa
    # path_to_events_file ='../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_lstmV2_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.02018-10-24 19:34:12'
    # avg + groundeR 
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankGroundRV2_fc_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.02018-10-24 15:39:41'
    # vlad + groundeR
    #path_to_events_file ='../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankGroundRV2_vlad_v1_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.0_centre_32_hidden_dim_1282018-10-24 15:39:21'
    # lstm + groundeR
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankGroundRV2_lstmV2_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_margin_10.02018-10-24 15:41:03'

    # lstm + dvsa + lamda 1.0
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_lstmV2_none_full_txt_lstmV2_rgb_i3d_lr_100.0_vid_lamda2_10.0_margin_10.02018-10-25 19:51:20'
    # att V1 lamda 1.0
    #path_to_events_file='../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_300_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_lamda2_10.0_margin_10.02018-10-17 16:28:40' 
    # att V1 lamda 0
    # path_to_events_file ='../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_300_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_margin_10.02018-10-18 19:24:30'
    # att V2 lamda 1.0
    # path_to_events_file ='../data/tensorBoardX/_bs_8_tn_30_wl_20_cn_1_fd_300_coAttV2_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_lamda2_10.0_margin_10.02018-10-20 14:03:49'
    # att V4, lamda 0
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_coAttV4_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_margin_10.02018-10-25 10:56:24'
    # att V4, lamda 1
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_300_coAttV4_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_lamda2_10.0_margin_10.02018-10-24 23:46:48'
   

    # att + lamda 1.0
    path_to_events_file='../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_coAtt_lstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_lamda2_10.0_margin_10.0lstm_hd_5122018-11-07 14:37:21'

    file_name_list_event = get_specific_file_list_from_fd(path_to_events_file, '.site', nameOnly=False)
    assert len(file_name_list_event)==1
    find_best_epoch_topK(file_name_list_event[0], epNum=30, topK=30)        
    print('######################################################################\n')
    pdb.set_trace()
    find_best_epoch_topK(file_name_list_event[0], epNum=30, \
            tag_name='Average_validation_accuracy', tag_check='Average_testing_accuracy', topK=5 )        

