CUDA_VISIBLE_DEVICES=0 python  ../fun/train.py --epSize 30 \
    --seed 0\
    --dbSet vid \
    --maxWordNum 20 --num_workers 3  --batchSize 3  --logFd  ../data/log/pm_resume_  \
    --outPre ../data/acc_grad --biLoss \
    --hdSize 300  --vwFlag  --stEp 0  --logFdTx ../data/tensorBoardX/pm_resume_ \
    --vis_dim 4096 --set_name train\
    --rpNum 30 --saveEp 1 \
    --txt_type gru --pos_type none \
    --lr 0.001 \
    --vis_ftr_type rgb_i3d \
    --margin 1 \
    --vis_type lstm \
    --visIter 5\
    --fc_feat_size 4096 \
    --dim_ftr 512  \
    --server_id 36\
    --stEp 0 \
    --optimizer sgd \
    --entropy_regu_flag --lamda2 1   \
    --wsMode coAttBi \
    --update_iter 4
    #--initmodel ../data/final_models/coAttV1_ep_18_lamda_1_ver.pth \
    #--loss_type  triplet_full  \
