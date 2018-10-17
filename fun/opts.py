#! encoding: UTF-8

import argparse
import sys


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--function', type=str, default='train')

    # activitynet path
    parser.add_argument('--act_train_json_path', type=str, default='data/activitynet/train.json')
    parser.add_argument('--act_val_1_json_path', type=str, default='data/activitynet/val_1.json')
    parser.add_argument('--act_val_2_json_path', type=str, default='data/activitynet/val_2.json')
    parser.add_argument('--act_test_ids_json_path', type=str, default='data/activitynet/test_ids.json')
    parser.add_argument('--act_val_ids_json_path', type=str, default='data/activitynet/val_ids.json')
    parser.add_argument('--act_train_ids_json_path', type=str, default='data/activitynet/train_ids.json')
    parser.add_argument('--act_data_info_path', type=str, default='data/activity_data_info.pkl')
    parser.add_argument('--act_feat_path', type=str, default='data/sub_activitynet_v1-3.c3d_rate_8.hdf5')


    parser.add_argument('--didemo_train_json_path', type=str, default='data/didemo/train_data.json')
    parser.add_argument('--didemo_val_json_path', type=str, default='data/didemo/val_data.json')
    parser.add_argument('--didemo_test_json_path', type=str, default='data/didemo/test_data.json')
    parser.add_argument('--didemo_feat_path', type=str, default='data/didemo/feats_5s')#ddm opt:data/didemo/feats_flownet2_tsn
    parser.add_argument('--didemo_feat_30s_path', type=str, default='')
    parser.add_argument('--didemo_data_info_path', type=str, default='data/didemo_data_info.pkl')

    parser.add_argument('--TACoS_data_info_path_v2_5', type=str, default='data/TACoS_data_info_v2_5.pkl')
    parser.add_argument('--TACoS_data_info_path_v2_1', type=str, default='data/TACoS_data_info_v2_1.pkl')
    parser.add_argument('--TACoS_data_info_path_v2_3', type=str, default='data/TACoS_data_info_v2_3.pkl')

    parser.add_argument('--TACoS_data_info_path_v3_5', type=str, default='data/TACoS_data_info_v3_5.pkl')
    parser.add_argument('--TACoS_data_info_path_v3_1', type=str, default='data/TACoS_data_info_v3_1.pkl')
    parser.add_argument('--TACoS_data_info_path_v3_3', type=str, default='data/TACoS_data_info_v3_3.pkl')

    parser.add_argument('--TACoS_data_info_5_path', type=str, default='data/TACoS_data_info_5.pkl')
    parser.add_argument('--TACoS_data_info_3_path', type=str, default='data/TACoS_data_info_3.pkl')
    parser.add_argument('--TACoS_data_info_1_path', type=str, default='data/TACoS_data_info_1.pkl')
    parser.add_argument('--TACoS_feat_path', type=str, default='data/tacos/feats')
    parser.add_argument('--TACoS_sample_rate', type=int, default=5)
    parser.add_argument('--TACoS_feat_mean_5s_path', type=str, default='data/tacos/feats_mean_5s')
    parser.add_argument('--TACoS_annos_fileName', type=str, default='data/tacos/TACoS_annos_fileName.mat')
    parser.add_argument('--TACoS_annos_sentence', type=str, default='data/tacos/TACoS_annos_sentence.mat')
    parser.add_argument('--TACoS_annos_sentenceProcessed', type=str, default='data/tacos/TACoS_annos_sentenceProcessed.mat')
    parser.add_argument('--TACoS_annos_startTimeSeconds', type=str, default='data/tacos/TACoS_annos_startTimeSeconds.mat')
    parser.add_argument('--TACoS_annos_endTimeSeconds', type=str, default='data/tacos/TACoS_annos_endTimeSeconds.mat')

    # anchor
    parser.add_argument('--act_n_anchors', type=int, default=90)
    parser.add_argument('--TACoS_n_anchors', type=int, default=60)
    
    parser.add_argument('--act_ground_truth_path', type=str, default='data/activity_ground_truth.pkl')
    parser.add_argument('--TACoS_ground_truth_path_v2_5', type=str, default='data/TACoS_ground_truth_v2_5.pkl')
    parser.add_argument('--TACoS_ground_truth_path_v2_1', type=str, default='data/TACoS_ground_truth_v2_1.pkl')
    parser.add_argument('--TACoS_ground_truth_path_v2_3', type=str, default='data/TACoS_ground_truth_v2_3.pkl')
    
    parser.add_argument('--TACoS_ground_truth_path_v3_5', type=str, default='data/TACoS_ground_truth_v3_5.pkl')
    parser.add_argument('--TACoS_ground_truth_path_v3_1', type=str, default='data/TACoS_ground_truth_v3_1.pkl')
    parser.add_argument('--TACoS_ground_truth_path_v3_3', type=str, default='data/TACoS_ground_truth_v3_3.pkl')

    parser.add_argument('--TACoS_ground_truth_5_path', type=str, default='data/TACoS_ground_truth_5.pkl')
    parser.add_argument('--TACoS_ground_truth_3_path', type=str, default='data/TACoS_ground_truth_3.pkl')
    parser.add_argument('--TACoS_ground_truth_1_path', type=str, default='data/TACoS_ground_truth_1.pkl')

    parser.add_argument('--didemo_ground_truth_path', type=str, default='data/didemo_ground_truth.pkl')
    parser.add_argument('--didemo_ground_truth_path_iou', type=str, default='data/didemo_ground_truth_iou.pkl')

    parser.add_argument('--word_count_threshold', type=int, default=5)
    parser.add_argument('--seed', type=int, default=110)
    #parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--video_time_step', type=int, default=300) #tacos300 DDM 6
    parser.add_argument('--caption_time_step', type=int, default=65) # tacos65 DDM 15
    parser.add_argument('--video_embedding_size', type=int, default=512)
    parser.add_argument('--word_embedding_size', type=int, default=512)
    parser.add_argument('--lstm_hidden_size', type=int, default=512)
    parser.add_argument('--att_hidden_size', type=int, default=512)


    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1)
    parser.add_argument('--learning_rate_decay_every', type=int, default=3)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)
    parser.add_argument('--optim_alpha', type=float, default=0.9)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    parser.add_argument('--optim_weight_decay', type=float, default=0.00001)


    parser.add_argument('--model_save_path', type=str, default='')
    parser.add_argument('--infer_model_path', type=str, default='')


    parser.add_argument('--feat_type', type=str, default='optical_flow')
    parser.add_argument('--fc_feat_size', type=int, default=1536) #inception 1536
    parser.add_argument('--fc_feat_path', type=str, default='')

    parser.add_argument('--topN1', type=int, default=1)
    parser.add_argument('--topN2', type=int, default=5)
    parser.add_argument('--iouM', type=float, default=0.1)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    opt = parse_opt()
    opt_dict = vars(opt)
    for k, v in opt_dict.items():
        print(k + ': \t' + str(v))
