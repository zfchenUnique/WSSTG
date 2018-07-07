from ../util.base_parser import BaseParser 

class stprParser(BaseParser):
    def __init__(self, *arg_list, **arg_dict):
        super(stprParser, self).__init__(*arg_list, **arg_dict)
        self.add_argument('--l1', default=1., type=float)
        self.add_argument('--l2', default=2., type=float)
        self.add_argument('--l3', default=0., type=float)
        self.add_argument('--l4', default=0.1, type=float)
        self.add_argument('--l5', default=0., type=float)
        self.add_argument('--dim_h1_tube', default=2048, type=int)
        self.add_argument('--dim_h1_desc', default=2048, type=int)
        self.add_argument('--dim_h2', default=512, type=int)
        self.add_argument('--n_pairs', default=50, type=int)
        self.add_argument('--initmodel', default='')
        self.add_argument('--resume', default='')
        self.add_argument('--batchsize', default=1500, type=int)
        self.add_argument('--gpu', default=0, type=int)
        self.add_argument('--decay', default=0.001, type=float)
        self.add_argument('--optimizer', default='adam', type=str)
        self.add_argument('--margin', default=0.1, type=float)
        self.add_argument('--feat_desc', default='hglmm_1000', type=str)
        self.add_argument('--feat_tube', default='mean', type=str)
        self.add_argument('--test',
                          action='store_true',
                          default=False)

def parse_args():
    parser = stprParser()
    args = parser.parse_args()
    return args
