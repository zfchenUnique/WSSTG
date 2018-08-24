import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import math


def getPosEmb(rpPropM, gFtrDim, kNN):
    """ Extract  geometric embedding for proposals

    Args:
        rpPropM: proposal list
        gFtrDim: geo feature dim
        kNN: number of neighbors

    Returns:
        position_matrix: [num_boxes, knn, gFtrDim]
    """
    rpPropMM = preprocessPrpList(rpPropM)
    position_matrix = extract_position_matrix(rpPropMM, nongt_dim=kNN)
    position_embedding = extract_position_embedding(position_matrix, feat_dim=gFtrDim)
    return position_embedding

def preprocessPrpList(rpPropList):
    """
    Args:
        nested rpPropList
    Returns:
        position_input: [bSize, frmSize, prpSize, ftrSize]
    """

    bSize = len(rpPropList)           
    frmSize = len(rpPropList[0])
    prpSize = len(rpPropList[0][0])
    ftrSize = len(rpPropList[0][0][0])
    rpList = list()
    for i in range(bSize):
        for ii in range(frmSize):
            for iii in range(prpSize):
                tmpMat = torch.FloatTensor(rpPropList[i][ii][iii])
                tmpMat[2] = tmpMat[0] + tmpMat[2]-1
                tmpMat[3] = tmpMat[1] + tmpMat[3]-1
                rpList.append(tmpMat)
    rpPropMM = torch.stack(rpList, 0).cuda()
    rpPropMM = rpPropMM.view(bSize, -1, ftrSize)
    #rpPropMM_reshpae = rpPropMM.view(bSize, frmSize, prpSize, ftrSize)
    return rpPropMM

def extract_position_matrix(bbox, nongt_dim):
    """ Extract position matrix

    Args:
        bbox: [bSize, num_boxes, 4]

    Returns:
        position_matrix: [bSize, num_boxes, nongt_dim, 4]
    """
    bSize, numBbx, dim = bbox.shape
    xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=2)
    # [bSize, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [bSize, num_boxes, num_boxes]
    center_x_exp = center_x.expand(bSize, numBbx, numBbx) 
    center_x_exp_trs = center_x_exp.transpose(1, 2)
    delta_x = center_x_exp -center_x_exp_trs 
    delta_x = torch.div(delta_x, bbox_width.transpose(1, 2))
    delta_x = torch.log(torch.clamp(torch.abs(delta_x), min=1e-3))
    
    center_y_exp = center_y.expand(bSize, numBbx, numBbx) 
    center_y_exp_trs = center_y_exp.transpose(1, 2)
    delta_y = center_y_exp -center_y_exp_trs 
    delta_y = torch.div(delta_y, bbox_height.transpose(1, 2))
    delta_y = torch.log(torch.clamp(torch.abs(delta_y), min=1e-3))
   
    bbox_width_exp = bbox_width.expand(bSize, numBbx, numBbx)
    delta_width = torch.div(bbox_width_exp, bbox_width.transpose(1, 2))
    delta_width = torch.log(delta_width)

    bbox_height_exp = bbox_height.expand(bSize, numBbx, numBbx)
    delta_height = torch.div(bbox_height_exp, bbox_height.transpose(1, 2))
    delta_height = torch.log(delta_height)

    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = sym[:, :, :nongt_dim]
        concat_list[idx] = sym.unsqueeze(3)
    position_matrix = torch.cat(concat_list, dim=3)
    return position_matrix

def extract_position_embedding(position_mat_raw, feat_dim, wave_length=1000):
    # position_mat, [bSize, num_rois, nongt_dim, 4]
    bSize, prpSize, kNN, dim = position_mat_raw.shape
    position_mat =  position_mat_raw.view(-1, position_mat_raw.shape[2], position_mat_raw.shape[3])
    feat_range = torch.arange(0, feat_dim / 8)
    dim_mat = torch.pow(wave_length, (8./feat_dim) * feat_range )
    dim_mat = dim_mat.view(1, 1, 1, -1).cuda()
    position_mat = 100.0 * position_mat.unsqueeze(3)
    div_mat = torch.div(position_mat, dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
    embedding = torch.cat((sin_mat, cos_mat), dim=3)
    # embedding, [bSize, num_rois, nongt_dim, feat_dim]
    embedding = embedding.view( bSize, prpSize, kNN, feat_dim)
    return embedding




class multiHeadAttention(nn.Module):
    def __init__(self, ftrDimList=[1024, 64, 1024], groupNum=16, kNN=None, atType='sp'):
        super(multiHeadAttention, self).__init__()
        visDim, geoDim, ftrOutDim = ftrDimList 
        assert visDim%groupNum==0, 'invalid group number'
        self.fcWQ = torch.nn.Linear(visDim, visDim, bias=False)
        self.fcWK = torch.nn.Linear(visDim, visDim, bias=False)
        self.fcWG = torch.nn.Linear(geoDim, groupNum, bias=False)
        self.grConca = torch.nn.Conv2d(visDim*groupNum, visDim, kernel_size=(1,1))
        self.atType = atType
        if kNN!=None:
            self.kNN = kNN
        else:
            self.kNN = 0 # change on the fly 
        self.visDim = visDim
        self.geoDim = geoDim
        self.ftrOutDim = ftrOutDim
        self.hdSize = visDim/groupNum
        self.groupNum =  groupNum

    def _initialize_weights(self):
        #pdb.set_trace()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, imFtrM, capFtrM=None, rpPropM=None):
        if self.atType =='sp' or self.atType=='spc':
            return self.forwardSp(imFtrM, rpPropM)

    def forwardSp(self, imFtrM, rpPropM):
        bSize, fSize, pSize, ftrSize = imFtrM.shape
#        pdb.set_trace()
        groupNum = self.groupNum
        self.kNN = fSize*pSize
        #pdb.set_trace()
        # [bSize, fSize*pSize, kNN, geoDim]  kNN = fSize*pSzie
        posEmb = getPosEmb(rpPropM, self.geoDim, self.kNN)     
        assert posEmb.shape[1]==posEmb.shape[2]
        # [bSize, -1, geoDim]
        posEmb_reshape = posEmb.view(bSize, -1, self.geoDim)
        # [bSize, -1, groupNum]
        posEmb_1 = self.fcWG(posEmb_reshape)
        posEmb_1_relu = F.relu(posEmb_1)
        # bSize, fSize*pSize, kNN, groupNum
        aff_weight = posEmb_1_relu.view(bSize, -1, self.kNN, self.groupNum)
        # bSize, fSize*pSize, groupNum, kNN
        aff_weight = torch.transpose(aff_weight, 2, 3)

        # feature embedding
        # bSize, -1, ftrSize
        imFtrM_reshape = imFtrM.view(bSize, -1, ftrSize)
        # bSize, -1, ftrSize
        q_data = self.fcWQ(imFtrM_reshape)
        # bSize, frmSize*prpSize, groupNum, -1
        q_data_batch = q_data.view(q_data.shape[0], q_data.shape[1], groupNum, -1) 
        # bSize, groupNum, frmSize*prpSize, -1
        q_data_batch = torch.transpose(q_data_batch, 1, 2)
        q_data_batch = q_data_batch.contiguous()

        # bSize, -1, ftrSize
        k_data = self.fcWK(imFtrM_reshape)
        # bSize, knn, groupNum, -1
        k_data_batch = k_data.view(k_data.shape[0], k_data.shape[1], groupNum, -1)
        # bSize, groupNum, frmSize*prpSize, -1
        k_data_batch = torch.transpose(k_data_batch, 1, 2)
        k_data_batch = k_data_batch.contiguous()
        #bSize, knn, ftrSize
        v_data = imFtrM_reshape
        #pdb.set_trace() 
        q_data_batch2 = q_data_batch.view(-1, q_data_batch.shape[2], q_data_batch.shape[3])
        k_data_batch2 = k_data_batch.view(-1, k_data_batch.shape[2], k_data_batch.shape[3])
        k_data_batch3 = k_data_batch2.permute(0, 2, 1)
        # bSize*groupNum, frmSize*prpSize, knn
        aff = torch.matmul(q_data_batch2,  k_data_batch3) 
        # bSize, groupNum,ftrSize*prpSize, knn
        aff_reshape = aff.view(bSize, groupNum, fSize*pSize, self.kNN)
        #aff_scale  = (1.0/math.sqrt(self.hdSize/groupNum))*aff_reshape
        #aff_scale  = 1000.0*aff_reshape
        aff_scale  = 100*aff_reshape
        # bSize, frmSize*prpSize, groupNum, knn
        aff_scale = aff_scale.transpose(1, 2)
        maxVal, maxIdx = torch.max(aff_scale, dim=3)
        #pdb.set_trace()
        # for numerical stability
        aff_scale = aff_scale - maxVal.unsqueeze(3).expand_as(aff_scale)

        weighted_aff = 0*torch.log(torch.clamp(aff_weight, min=1e-6)) + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=3)
        # bSize, frmSize*prpSize*groupNum, knn 
        aff_softmax_reshape = aff_softmax.view(bSize,  -1, self.kNN)
        # bSize, frmSize*prpSize*groupNum, ftrDim
        output_t = torch.bmm(aff_softmax_reshape, v_data)
        # bSize*frmSize*prpSize, groupNum*ftrDim,1 , 1
        output_t_reshape =  output_t.view(bSize*fSize*pSize, groupNum*ftrSize, 1, 1)
        # bSize*frmSize*prpSize, groupNum, ftrDim, 1
        linear_out = self.grConca(output_t_reshape)
        linear_out_reshape = linear_out.view(bSize, fSize, pSize, ftrSize)
        #return linear_out_reshape
        # debug visualize relations
        return linear_out_reshape, aff_softmax, aff_scale, torch.clamp(aff_weight, min=1e-6)
