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

def find_best_epoch(file_name, epNum=30, tag_name='Average_testing_accuracy', tag_check='Average_validation_accuracy', ep_itr=5437):
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
    print('epoch id %f\n' %(best_ele.step*1.0/ep_itr))
    print('finish!\n')

def find_best_epoch_topK(file_name, epNum=30, tag_name='Average_testing_accuracy', tag_check='Average_validation_accuracy', ep_itr=5437, topK=1):
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
    print('epoch id %f\n' %(best_check_ele.step*1.0/ep_itr))
    print('finish!\n')
    pdb.set_trace()

if __name__=='__main__':
    # dvsa lstm
    path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_lstmV2_none_full_txt_lstmV2_i3d_lr_100.0_actNet_margin_10.02018-10-25 23:15:20'
    # dvsa average
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_fc_none_full_txt_lstmV2_i3d_lr_100.0_actNet_margin_10.02018-10-26 10:23:28'
    # dvsa vlad
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankTube_vlad_v1_none_full_txt_lstmV2_i3d_lr_100.0_actNet_margin_10.0_centre_32_hidden_dim_1282018-10-26 10:58:35'
    # groundeR avg
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankGroundRV2_fc_none_full_txt_lstmV2_i3d_lr_100.0_actNet_margin_10.02018-10-29 10:20:53'
    # groundeR lstm
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankGroundRV2_lstmV2_none_full_txt_lstmV2'
    # groundeR vlad
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_512_rankGroundRV2_vlad_v1_none_full_txt_lstmV2_i3d_lr_100.0_actNet_margin_10.0_centre_32_hidden_dim_1282018-10-27 16:35:16'
    # coAtt , lamda 0
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_300_coAtt_lstm_none_full_txt_gru_i3d_lr_100.0_actNet_margin_10.02018-10-09 17:04:01'
    # coAtt , lamda 1
    #path_to_events_file = '../data/tensorBoardX/_bs_16_tn_30_wl_20_cn_1_fd_300_coAtt_lstm_none_full_txt_gru_i3d_lr_100.0_actNet_lamda2_10.0_margin_10.02018-10-09 16:47:46'
    
    file_name_list_event = get_specific_file_list_from_fd(path_to_events_file, '.site', nameOnly=False)
    assert len(file_name_list_event)==1
    find_best_epoch_topK(file_name_list_event[0], epNum=300, \
            tag_name='Average_validation_accuracy', tag_check='Average_testing_accuracy', topK=5 )        

