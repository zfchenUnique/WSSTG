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
        self.add_argument('--vocaSize', default=1312, type=int)
        self.add_argument('--conSecFlag', action='store_true', default=False)
        self.add_argument('--conFrmNum', default=9, type=int)
        self.add_argument('--moduleNum', default=2, type=int)
        self.add_argument('--moduleHdSize', default=1024, type=int)
        self.add_argument('--stEp', default=0, type=int)
        self.add_argument('--keepKeyFrameOnly', action='store_true', default=False)
        self.add_argument('--visRsFd', default='../data/visResult/rank_', type=str)
        self.add_argument('--logFdTx', default='../data/log/wsEmb', type=str)
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

def parse_args():
    parser = wsParamParser()
    args = parser.parse_args()
    half_size ='full'
    if args.half_size:
        half_size = 'half'
    struct_ann = ''
    if args.struct_flag:
        struct_ann = '_struct_ann_lamda_%d' %(int(args.lamda*10))

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
    return args
