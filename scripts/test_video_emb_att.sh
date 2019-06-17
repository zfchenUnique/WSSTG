CUDA_VISIBLE_DEVICES=0 python  ../fun/eval.py --epSize 1000 \
    --dbSet vid \
    --maxWordNum 20 --num_workers 0  --batchSize 1  --logFd  ../data/final_models/abs_  \
    --outPre ../data/vd_model/ --biLoss \
    --hdSize 300  --vwFlag  --stEp 0  --logFdTx ../data/tensorBoardX/ \
    --vis_dim 4096 --set_name val\
    --rpNum 30 --saveEp 1 \
    --txt_type gru --pos_type none \
    --lr 0.001 \
    --vis_ftr_type rgb_i3d \
    --margin 1 \
    --vis_type lstm \
    --visIter 1\
    --server_id 36\
    --wsMode coAtt \
    --fc_feat_size 4096 \
    --dim_ftr 512  \
    --no_shuffle_flag \
    --eval_test_flag \
    --initmodel ../data/final_models/coAttV1_ep_18_lamda_1_ver.pth \
    #--initmodel ../data/final_models/coAttV1_ep_21_lamda_1.pth 
    #--initmodel ../data/final_models/coAttV1_ep_18_lamda_1_ver.pth \
    #--initmodel ../data/final_models/coAttV1_ep_21_lamda_1.pth 
    #--initmodel ../data/vd_model/verify_tanh_1000_bs_16_tn_30_wl_20_cn_1_fd_512_coAttlstm_none_full_txt_gru_rgb_i3d_lr_100.0_vid_lamda2_10.0_margin_10.0lstm_hd_512/_ep_1_itr_0.pth
    #--initmodel ../data/final_models/coAttV1_ep_21_lamda_1.pth \
    #--eval_test_flag \
    #--initmodel ../data/final_models/coAttV1_ep_7_lamda_0.pth \
    #--initmodel ../data/final_models/coAttV1_ep_7_lamda_0.pth \
    #--eval_val_flag \
    #--initmodel ../data/final_models/coAttV1_ep_18_lamda_1_ver.pth \
