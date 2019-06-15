import sys
sys.path.append('..')
from util.base_parser import BaseParser 

class wsParamParser(BaseParser):
    def __init__(self, *arg_list, **arg_dict):
        super(wsParamParser, self).__init__(*arg_list, **arg_dict)
        self.add_argument('--l1', default=1., type=float)
        self.add_argument('--l2', default=2., type=float)
        self.add_argument('--dim_ftr', default=128, type=int)
        self.add_argument('--n_pairs', default=50, type=int)
        self.add_argument('--initmodel', default=None)
        self.add_argument('--resume', default='')
        self.add_argument('--batchSize', default=16, type=int)
        self.add_argument('--gpu', default=1, type=int)
        self.add_argument('--decay', default=0.001, type=float)
        self.add_argument('--optimizer', default='sgd', type=str)
        self.add_argument('--margin', default=1, type=float)
        self.add_argument('--test', action='store_true', default=False)
        self.add_argument('--dbSet', default='otb', type=str)
        self.add_argument('--num_workers', default=8, type=int)
        self.add_argument('--visIter', default=5, type=int)
        self.add_argument('--evalIter', default=10, type=int)
        self.add_argument('--epSize', default=10000, type=int)
        self.add_argument('--logFd', default='../data/log/wsEmb', type=str)
        self.add_argument('--saveEp', default=10, type=int)
        self.add_argument('--outPre', default='../data/models/a2d_checkpoint_', type=str)
        self.add_argument('--biLoss', action='store_true', default=False)
        self.add_argument('--lossW', action='store_true', default=False)
        self.add_argument('--lamda', default=0.8, type=float)
        self.add_argument('--vwFlag', action='store_true', default=False)
        self.add_argument('--wsMode', default='rank', type=str)
        self.add_argument('--hdSize', default=128, type=int)
        self.add_argument('--vocaSize', default=1900, type=int)
        self.add_argument('--conSecFlag', action='store_true', default=False)
        self.add_argument('--conFrmNum', default=9, type=int)
        self.add_argument('--moduleNum', default=2, type=int)
        self.add_argument('--moduleHdSize', default=1024, type=int)
        self.add_argument('--stEp', default=0, type=int)
        self.add_argument('--keepKeyFrameOnly', action='store_true', default=False)
        self.add_argument('--visRsFd', default='../data/visResult/rank_', type=str)
        self.add_argument('--logFdTx', default='../logs/wsEmb', type=str)
        self.add_argument('--set_name', default='train', type=str)
        self.add_argument('--isParal', action='store_true', default=False)
        self.add_argument('--capNum', default=1, type=int)
        self.add_argument('--maxWordNum', default=20, type=int)
        self.add_argument('--rpNum', default=30, type=int)
        self.add_argument('--vis_dim', default=2048, type=int)
        self.add_argument('--vis_type', default='lstm', type=str)
        self.add_argument('--txt_type', default='lstm', type=str)
        self.add_argument('--pos_type', default='aiayn', type=str)
        self.add_argument('--pos_emb_dim', default=64, type=int)
        self.add_argument('--half_size', action='store_true', default=False)
        self.add_argument('--server_id', default=36, type=int)
        self.add_argument('--vis_ftr_type', default='rgb', type=str)
        self.add_argument('--struct_flag', action='store_true', default=False)
        self.add_argument('--struct_only', action='store_true', default=False)
        self.add_argument('--eval_val_flag', action='store_true', default=False)
        self.add_argument('--eval_test_flag', action='store_true', default=False)
        self.add_argument('--entropy_regu_flag', action='store_true', default=False)
        self.add_argument('--lamda2', default=0.1, type=float)
        self.add_argument('--hidden_dim', default=128, type=int)
        self.add_argument('--centre_num', default=32, type=int)
        self.add_argument('--vlad_alpha', default=1.0, type=float)
        self.add_argument('--cache_flag', action='store_true', default=False)
        self.add_argument('--use_mean_cache_flag', action='store_true', default=False)
        self.add_argument('--dropout_prob', type=float, default=0.1)
        self.add_argument('--batch_size', type=int, default=64)
        self.add_argument('--video_time_step', type=int, default=20) #tacos300 DDM 6
        self.add_argument('--caption_time_step', type=int, default=20) # tacos65 DDM 15
        self.add_argument('--video_embedding_size', type=int, default=512)
        self.add_argument('--fc_feat_size', type=int, default=2048)
        self.add_argument('--word_embedding_size', type=int, default=512)
        self.add_argument('--lstm_hidden_size', type=int, default=512)
        self.add_argument('--att_hidden_size', type=int, default=512)
        self.add_argument('--n_anchors', type=int, default=1)
        self.add_argument('--word_cnt', type=int, default=20)
        self.add_argument('--context_flag', action='store_true', default=False)
        self.add_argument('--no_shuffle_flag', action='store_true', default=False)
        self.add_argument('--frm_level_flag', action='store_true', default=False)
        self.add_argument('--frm_num', type=int, default=1)
        self.add_argument('--att_exp', type=int, default=1)
        self.add_argument('--loss_type', default='triplet_mil', type=str)
        self.add_argument('--use_gt_region', action='store_true', default=False)
        self.add_argument('--seed', default=0, type=int)
        self.add_argument('--update_iter', default=4, type=int)


def parse_args():
    parser = wsParamParser()
    args = parser.parse_args()
    half_size ='full'
    if args.half_size:
        half_size = 'half'
    struct_ann = ''
    if args.struct_flag:
        struct_ann = '_struct_ann_lamda_%d' %(int(args.lamda*10))
        if args.struct_only:
            struct_ann =  struct_ann + '_only'
    if args.entropy_regu_flag:
        struct_ann = struct_ann + '_lamda2_' + str(args.lamda2*10)

    struct_ann + args.loss_type

    struct_ann = struct_ann + '_margin_'+ str(args.margin*10)+ '_att_exp' + str(args.att_exp)

    if args.vis_type =='vlad_v1':
        struct_ann = struct_ann + '_centre_' + str(args.centre_num) \
                + '_hidden_dim_' + str(args.hidden_dim)
    
    if args.context_flag:
        struct_ann = struct_ann + '_context'

    if args.wsMode == 'coAtt' or args.wsMode == 'coAttGroundR' or args.wsMode == 'coAttBi':
        struct_ann = struct_ann + 'lstm_hd_' + str(args.lstm_hidden_size) +'_seed_' + str(args.seed) 

    if args.frm_level_flag:
        struct_ann = struct_ann + '_frm_level_'

    if args.lossW:
        struct_ann = struct_ann + 'weak_weight_'+str(args.lamda*10)



    args.logFd = args.logFd +'_bs_'+str(args.batchSize) + '_tn_' + str(args.rpNum) \
            +'_wl_' +str(args.maxWordNum) + '_cn_' + str(args.capNum) +'_fd_'+ str(args.dim_ftr) \
            + '_' + str(args.wsMode) +'_' +str(args.vis_type)+ '_' + str(args.pos_type) + \
            '_' + half_size + '_txt_' + str(args.txt_type) + '_' + str(args.vis_ftr_type) \
            + '_lr_' + str(args.lr*100000) + '_' + str(args.dbSet) + struct_ann
    
    args.outPre = args.outPre +'_bs_'+str(args.batchSize) + '_tn_' + str(args.rpNum) \
            +'_wl_'+str(args.maxWordNum) + '_cn_' + str(args.capNum) +'_fd_'+ str(args.dim_ftr) \
             + '_' + str(args.wsMode) + str(args.vis_type) + '_'+ str(args.pos_type) + \
             '_'+ half_size+'_txt_'+str(args.txt_type)+ '_' +str(args.vis_ftr_type)  + \
             '_lr_' + str(args.lr*100000) + '_' + str(args.dbSet) + struct_ann +'/'
    
    args.logFdTx = args.logFdTx +'_bs_'+str(args.batchSize) + '_tn_' + str(args.rpNum) \
            +'_wl_' +str(args.maxWordNum) + '_cn_' + str(args.capNum) +'_fd_'+ str(args.dim_ftr) \
            + '_' + str(args.wsMode) +'_' +str(args.vis_type)+ '_' + str(args.pos_type) +'_' + \
            half_size +'_txt_'+ str(args.txt_type) + '_' + str(args.vis_ftr_type) + '_lr_' + \
            str(args.lr*100000) + '_' + str(args.dbSet) + struct_ann
    args.visRsFd = args.visRsFd + args.dbSet + '_'
    return args
