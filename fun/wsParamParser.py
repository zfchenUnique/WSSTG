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
        self.add_argument('--batchSize', default=17, type=int)
        self.add_argument('--gpu', default=1, type=int)
        self.add_argument('--decay', default=0.001, type=float)
        self.add_argument('--optimizer', default='sgd', type=str)
        self.add_argument('--margin', default=1, type=float)
        self.add_argument('--test', action='store_true', default=False)
        self.add_argument('--dbSet', default='otb', type=str)
        self.add_argument('--num_workers', default=16, type=int)
        self.add_argument('--k_img', default=1, type=int)
        self.add_argument('--k_prp', default=20, type=int)
        self.add_argument('--maxWL', default=15, type=int)
        self.add_argument('--visIter', default=1, type=int)
        self.add_argument('--evalIter', default=10, type=int)
        self.add_argument('--epSize', default=10000, type=int)
        self.add_argument('--logFd', default='../data/log/wsEmb', type=str)
        self.add_argument('--saveEp', default=10, type=int)
        self.add_argument('--outPre', default='../data/models/a2d_checkpoint_', type=str)

def parse_args():
    parser = wsParamParser()
    args = parser.parse_args()
    return args
