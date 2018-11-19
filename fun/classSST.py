# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ipdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import opts
from classLSTMCore import LSTMCore
import pdb


class SST(nn.Module):
    def __init__(self, opt):
        super(SST, self).__init__()

        self.word_cnt = opt.word_cnt
        self.fc_feat_size = opt.fc_feat_size
        self.video_embedding_size = opt.video_embedding_size
        self.word_embedding_size = opt.word_embedding_size
        self.lstm_hidden_size = opt.lstm_hidden_size
        self.video_time_step = opt.video_time_step
        self.caption_time_step = opt.caption_time_step
        self.dropout_prob = opt.dropout_prob
        self.n_anchors = opt.n_anchors
        self.att_hidden_size = opt.att_hidden_size

        self.video_embedding = nn.Linear(self.fc_feat_size, self.video_embedding_size)

        #self.lstm_video = LSTMCore(self.video_embedding_size, self.lstm_hidden_size, self.dropout_prob)
        #self.lstm_caption = LSTMCore(self.word_embedding_size, self.lstm_hidden_size, self.dropout_prob)
        self.lstm_video = torch.nn.LSTMCell(self.video_embedding_size, self.lstm_hidden_size, bias=True)
        self.lstm_caption = torch.nn.LSTMCell(self.word_embedding_size, self.lstm_hidden_size, bias=True)


        self.vid_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        self.sen_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        # nn.Tanh 是一种不带任何参数的 operation，所以直接在 forward 里面调用 F.tanh 即可
        # self.tanh_att = nn.Tanh()

        self.att_linear = nn.Linear(self.att_hidden_size, 1)

        self.h2o = nn.Linear(self.lstm_hidden_size * 2, self.n_anchors)

        # self.softmax_att = nn.Softmax()

        # 注意因为 idx_to_word 是从 1 开始, 此处要加 1, 要不然会遇到 bug:
        # cuda runtime error (59): device-side assert triggered
        #self.word_embedding = nn.Embedding(self.word_cnt + 1, self.word_embedding_size)
        self.word_embedding = nn.Linear(300, self.word_embedding_size)
        # self.vocab_logit = nn.Linear(self.lstm_hidden_size, self.word_cnt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.video_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_embedding.bias.data.fill_(0)

        self.word_embedding.weight.data.uniform_(-initrange, initrange)
        self.word_embedding.bias.data.fill_(0)

        # self.vocab_logit.weight.data.uniform_(-initrange, initrange)
        # self.vocab_logit.bias.data.fill_(0)

        self.vid_linear.weight.data.uniform_(-initrange, initrange)
        self.vid_linear.bias.data.fill_(0)

        self.sen_linear.weight.data.uniform_(-initrange, initrange)
        self.sen_linear.bias.data.fill_(0)

        self.att_linear.weight.data.uniform_(-initrange, initrange)
        self.att_linear.bias.data.fill_(0)

        self.h2o.weight.data.uniform_(-initrange, initrange)
        self.h2o.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state

    def init_hidden_new(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state



    # video_fc_feats: batch * encode_time_step * fc_feat_size
    def forward(self, video_fc_feat, video_caption, cap_length_list=None):
        #pdb.set_trace()
        batch_size = video_fc_feat.size(0)
        batch_size_caption = video_caption.size(0)

        video_state = self.init_hidden_new(batch_size)
        caption_state = self.init_hidden_new(batch_size_caption)

        # 将 caption 用 LSTM 进行编码, 用来 soft attention
        caption_outputs = []
        caption_time_step = video_caption.size(1)
        for i in range(caption_time_step):
            word = video_caption[:, i].clone()
            if video_caption[:, i].data.sum() == 0:
                break
            #import ipdb
            #pdb.set_trace()
            caption_xt = self.word_embedding(word)
            #caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_outputs.append(caption_output)
            caption_state = (caption_output, caption_state)
        # caption_outputs: batch * caption_time_step * lstm_hidden_size
        caption_outputs = torch.cat([_.unsqueeze(1) for _ in caption_outputs], 1).contiguous()

        # 将 video fc feat 用 LSTM 进行编码
        video_outputs = []
        for i in range(self.video_time_step):
            video_xt = self.video_embedding(video_fc_feat[:, i, :])
            video_output, video_state = self.lstm_video.forward(video_xt, video_state)
            video_outputs.append(video_output)
            video_state = (video_output, video_state)
        # video_outputs: batch * video_time_step * lstm_hidden_size
        video_outputs = torch.cat([_.unsqueeze(1) for _ in video_outputs], 1).contiguous()

        # soft attention for caption based on each video
        output_list = list()
        for i in range(self.video_time_step):
            # part 1
            video_outputs_linear = self.vid_linear(video_outputs[:, i, :])
            video_outputs_linear_expand = video_outputs_linear.expand(caption_outputs.size(1), video_outputs_linear.size(0),
                                                                      video_outputs_linear.size(1)).transpose(0, 1)

            # part 2
            caption_outputs_flatten = caption_outputs.view(-1, self.lstm_hidden_size)
            caption_outputs_linear = self.sen_linear(caption_outputs_flatten)
            caption_outputs_linear = caption_outputs_linear.view(batch_size_caption, caption_outputs.size(1), self.att_hidden_size)

            # part 1 and part 2 attention
            sig_probs = []
            for cap_id in range(batch_size_caption):
                #pdb.set_trace()
                cap_length = max(cap_length_list[cap_id], 1)
                caption_output_linear_cap_id = caption_outputs_linear[cap_id, : cap_length, :]
                video_outputs_linear_expand_clip = video_outputs_linear_expand[:, :cap_length, :]
                caption_outputs_linear_cap_id_exp = caption_output_linear_cap_id.expand_as(video_outputs_linear_expand_clip) 
                video_caption = F.tanh(video_outputs_linear_expand_clip \
                        + caption_outputs_linear_cap_id_exp*0)*1000
                #video_caption = F.tanh(video_outputs_linear_expand_clip \
                #        + caption_outputs_linear_cap_id_exp)
                #video_caption = F.relu(video_outputs_linear_expand_clip \
                #        + caption_outputs_linear_cap_id_exp)
                
                #video_caption = video_outputs_linear_expand_clip \
                #        + caption_outputs_linear_cap_id_exp
                
                video_caption_view = video_caption.contiguous().view(-1, self.att_hidden_size)
                video_caption_out = self.att_linear(video_caption_view)
                video_caption_out_view = video_caption_out.view(-1, cap_length)
                #print(torch.min(video_caption_out_view))
                atten_weights = nn.Softmax(dim=1)(video_caption_out_view).unsqueeze(2)
                #print(torch.min(atten_weights))
                if i==0 or i==5 or i== 14:
                    aa, bb= torch.sort(video_caption_out_view)
                    print(atten_weights[1])
                    print(video_caption_out_view[1])
                    print(bb[1])
                    pdb.set_trace()

                # http://pytorch.org/docs/master/torch.html#torch.bmm
                # batch 1: b x n x m, batch 2: b x m x p, 结果为: b x n x p
                #aa, bb= torch.sort(video_caption_out_view)
                #print(bb[0])
                #print(aa[0])
                #print(aa[1])
                #pdb.set_trace()
                caption_output_cap_id = caption_outputs[cap_id, : cap_length, :]
                caption_output_cap_id_exp = caption_output_cap_id.expand(batch_size,\
                        caption_output_cap_id.size(0), caption_output_cap_id.size(1))
                atten_caption = torch.bmm(caption_output_cap_id_exp.transpose(1, 2), atten_weights).squeeze(2)

                video_caption_hidden = torch.cat((atten_caption, video_outputs[:, i, :]), dim=1)
                #cur_probs = self.h2o(video_caption_hidden)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #cur_probs = F.sigmoid(self.h2o(video_caption_hidden))
                cur_probs = cos(atten_caption, video_outputs[: ,i, :]).unsqueeze(1)
                #pdb.set_trace()
                #cur_probs = torch.sigmoid(cur_probs)


                sig_probs.append(cur_probs)

            sig_probs = torch.cat([_ for _ in sig_probs], 1).contiguous()
            output_list.append(sig_probs)
        simMM = torch.stack(output_list, dim=2).mean(2)
        #print(torch.max(simMM))
        #print(torch.min(simMM))
        #sig_probs = torch.cat([_.unsqueeze(1) for _ in sig_probs], 1).contiguous()
        #sig_probs = nn.Softmax()(sig_probs.view(batch_size, -1)).view(batch_size,video_outputs.size(1),-1)
        #print(video_caption_out_view[0, ...])
        #print(atten_weights[0,...])
        return simMM

# SST + Ground-R

class SSTGroundR(nn.Module):
    def __init__(self, opt):
        super(SSTGroundR, self).__init__()

        self.word_cnt = opt.word_cnt
        self.fc_feat_size = opt.fc_feat_size
        self.video_embedding_size = opt.video_embedding_size
        self.word_embedding_size = opt.word_embedding_size
        self.lstm_hidden_size = opt.lstm_hidden_size
        self.video_time_step = opt.video_time_step
        self.caption_time_step = opt.caption_time_step
        self.dropout_prob = opt.dropout_prob
        self.n_anchors = opt.n_anchors
        self.att_hidden_size = opt.att_hidden_size

        self.video_embedding = nn.Linear(self.fc_feat_size, self.video_embedding_size)

        #self.lstm_video = LSTMCore(self.video_embedding_size, self.lstm_hidden_size, self.dropout_prob)
        #self.lstm_caption = LSTMCore(self.word_embedding_size, self.lstm_hidden_size, self.dropout_prob)
        self.lstm_video = torch.nn.LSTMCell(self.video_embedding_size, self.lstm_hidden_size, bias=True)
        self.lstm_caption = torch.nn.LSTMCell(self.word_embedding_size, self.lstm_hidden_size, bias=True)


        self.vid_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        self.sen_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        # nn.Tanh 是一种不带任何参数的 operation，所以直接在 forward 里面调用 F.tanh 即可
        # self.tanh_att = nn.Tanh()

        self.att_linear = nn.Linear(self.att_hidden_size, 1)

        self.h2o = nn.Linear(self.lstm_hidden_size * 2, self.n_anchors)

        # self.softmax_att = nn.Softmax()

        # 注意因为 idx_to_word 是从 1 开始, 此处要加 1, 要不然会遇到 bug:
        # cuda runtime error (59): device-side assert triggered
        #self.word_embedding = nn.Embedding(self.word_cnt + 1, self.word_embedding_size)
        self.word_embedding = nn.Linear(300, self.word_embedding_size)
        # self.vocab_logit = nn.Linear(self.lstm_hidden_size, self.word_cnt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.video_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_embedding.bias.data.fill_(0)

        self.word_embedding.weight.data.uniform_(-initrange, initrange)
        self.word_embedding.bias.data.fill_(0)

        # self.vocab_logit.weight.data.uniform_(-initrange, initrange)
        # self.vocab_logit.bias.data.fill_(0)

        self.vid_linear.weight.data.uniform_(-initrange, initrange)
        self.vid_linear.bias.data.fill_(0)

        self.sen_linear.weight.data.uniform_(-initrange, initrange)
        self.sen_linear.bias.data.fill_(0)

        self.att_linear.weight.data.uniform_(-initrange, initrange)
        self.att_linear.bias.data.fill_(0)

        self.h2o.weight.data.uniform_(-initrange, initrange)
        self.h2o.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state

    def init_hidden_new(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state



    # video_fc_feats: batch * encode_time_step * fc_feat_size
    def forward(self, video_fc_feat, video_caption, cap_length_list=None):
        batch_size = video_fc_feat.size(0)
        batch_size_caption = video_caption.size(0)
        rp_num = int(batch_size/batch_size_caption)  

        video_state = self.init_hidden_new(batch_size)
        caption_state = self.init_hidden_new(batch_size_caption)

        # 将 caption 用 LSTM 进行编码, 用来 soft attention
        caption_outputs = []
        caption_time_step = video_caption.size(1)
        for i in range(caption_time_step):
            word = video_caption[:, i].clone()
            if video_caption[:, i].data.sum() == 0:
                break
            #import ipdb
            #pdb.set_trace()
            caption_xt = self.word_embedding(word)
            #caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_outputs.append(caption_output)
            caption_state = (caption_output, caption_state)
        # caption_outputs: batch * caption_time_step * lstm_hidden_size
        caption_outputs = torch.cat([_.unsqueeze(1) for _ in caption_outputs], 1).contiguous()

        # 将 video fc feat 用 LSTM 进行编码
        video_outputs = []
        for i in range(self.video_time_step):
            video_xt = self.video_embedding(video_fc_feat[:, i, :])
            video_output, video_state = self.lstm_video.forward(video_xt, video_state)
            video_outputs.append(video_output)
            video_state = (video_output, video_state)
        # video_outputs: batch * video_time_step * lstm_hidden_size
        video_outputs = torch.cat([_.unsqueeze(1) for _ in video_outputs], 1).contiguous()

        # soft attention for caption based on each video
        output_list = list()
        
        caption_outputs_flatten = caption_outputs.view(-1, self.lstm_hidden_size)
        caption_outputs_linear = self.sen_linear(caption_outputs_flatten)
        caption_outputs_linear = caption_outputs_linear.view(batch_size_caption, 1, caption_outputs.size(1), self.att_hidden_size)
        #pdb.set_trace()
        caption_outputs_linear_exp = caption_outputs_linear.expand(batch_size_caption, rp_num, caption_outputs.size(1)\
                , self.att_hidden_size).contiguous()
        # (b*rp_num), t_c, d_c
        caption_outputs_linear_exp_view = caption_outputs_linear_exp.view(batch_size, caption_time_step, self.att_hidden_size)

        output_list = list() 
        output_cap_ftr_list = list()
        for i in range(self.video_time_step):
            # part 1
            # (b*rp_num), d_v
            video_outputs_linear = self.vid_linear(video_outputs[:, i, :])
            # (b*rp_num), t_c ,d_v
            video_outputs_linear_expand = video_outputs_linear.expand(caption_outputs.size(1), video_outputs_linear.size(0),
                                                                      video_outputs_linear.size(1)).transpose(0, 1)
            
            video_caption = F.tanh(video_outputs_linear_expand \
                        + caption_outputs_linear_exp_view)
            video_caption_view = video_caption.contiguous().view(-1, self.att_hidden_size)
            # b*rp_num, t_c 
            video_caption_out = self.att_linear(video_caption_view)
            #pdb.set_trace()
            # b*rp_num, t_c 
            video_caption_out_view = video_caption_out.view(batch_size_caption, rp_num, caption_time_step)
            # b, rp_num, d_v
            tmp_video_outputs = video_outputs[:, i, :].view(batch_size_caption, rp_num, -1)
            
            tmp_cap_ftr_list = list()
            for cap_id in range(batch_size_caption):
                cap_length = max(cap_length_list[cap_id], 1)
                # rp_num, t_c 
                atten_weights = nn.Softmax(dim=1)(video_caption_out_view[cap_id, :, :cap_length]).unsqueeze(2)
                caption_output_cap_id = caption_outputs[cap_id, : cap_length, :]
                # rp_num, t_c, d_c
                caption_output_cap_id_exp = caption_output_cap_id.expand(rp_num,\
                        caption_output_cap_id.size(0), caption_output_cap_id.size(1))
                # rp_num , d_c, 1
                atten_caption = torch.bmm(caption_output_cap_id_exp.transpose(1, 2), atten_weights).squeeze(2)
                
                video_caption_hidden = torch.cat((atten_caption, tmp_video_outputs[cap_id]), dim=1)
                tmp_cap_ftr_list.append(atten_caption)
                #cur_probs = self.h2o(video_caption_hidden)
                #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                # rp_num *1
                #cur_probs = cos(atten_caption, tmp_video_outputs[cap_id]).unsqueeze(0)
                #sig_probs.append(cur_probs)
                 
            # b, rp_num, d_c
            cap_ftr_mat = torch.stack(tmp_cap_ftr_list, dim=0).contiguous()
            output_cap_ftr_list.append(cap_ftr_mat)
        output_cap_ftr_mat = torch.stack(output_cap_ftr_list, dim=3)
        output_cap_ftr_mat_mean = output_cap_ftr_mat.mean(3)
        # b, rp_num, d_v
        video_output_mean = video_outputs.mean(1).view(batch_size_caption, rp_num, -1)
        return video_output_mean, output_cap_ftr_mat_mean


class SSTMul(nn.Module):
    def __init__(self, opt):
        super(SSTMul, self).__init__()

        self.word_cnt = opt.word_cnt
        self.fc_feat_size = opt.fc_feat_size
        self.video_embedding_size = opt.video_embedding_size
        self.word_embedding_size = opt.word_embedding_size
        self.lstm_hidden_size = opt.lstm_hidden_size
        self.video_time_step = opt.video_time_step
        self.caption_time_step = opt.caption_time_step
        self.dropout_prob = opt.dropout_prob
        self.n_anchors = opt.n_anchors
        self.att_hidden_size = opt.att_hidden_size

        self.video_embedding = nn.Linear(self.fc_feat_size, self.video_embedding_size)

        #self.lstm_video = LSTMCore(self.video_embedding_size, self.lstm_hidden_size, self.dropout_prob)
        #self.lstm_caption = LSTMCore(self.word_embedding_size, self.lstm_hidden_size, self.dropout_prob)
        self.lstm_video = torch.nn.LSTMCell(self.video_embedding_size, self.lstm_hidden_size, bias=True)
        self.lstm_caption = torch.nn.LSTMCell(self.word_embedding_size, self.lstm_hidden_size, bias=True)


        self.vid_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        self.sen_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        
        self.vid_linear_v2 = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        self.sen_linear_v2 = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        # nn.Tanh 是一种不带任何参数的 operation，所以直接在 forward 里面调用 F.tanh 即可
        # self.tanh_att = nn.Tanh()

        self.att_linear = nn.Linear(self.att_hidden_size, 1)
        self.att_linear_v2 = nn.Linear(self.att_hidden_size, 1)

        self.h2o = nn.Linear(self.lstm_hidden_size * 2, self.n_anchors)

        # self.softmax_att = nn.Softmax()

        # 注意因为 idx_to_word 是从 1 开始, 此处要加 1, 要不然会遇到 bug:
        # cuda runtime error (59): device-side assert triggered
        #self.word_embedding = nn.Embedding(self.word_cnt + 1, self.word_embedding_size)
        self.word_embedding = nn.Linear(300, self.word_embedding_size)
        # self.vocab_logit = nn.Linear(self.lstm_hidden_size, self.word_cnt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.video_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_embedding.bias.data.fill_(0)

        self.word_embedding.weight.data.uniform_(-initrange, initrange)
        self.word_embedding.bias.data.fill_(0)

        # self.vocab_logit.weight.data.uniform_(-initrange, initrange)
        # self.vocab_logit.bias.data.fill_(0)

        self.vid_linear.weight.data.uniform_(-initrange, initrange)
        self.vid_linear.bias.data.fill_(0)

        self.sen_linear.weight.data.uniform_(-initrange, initrange)
        self.sen_linear.bias.data.fill_(0)

        self.att_linear.weight.data.uniform_(-initrange, initrange)
        self.att_linear.bias.data.fill_(0)


        self.vid_linear_v2.weight.data.uniform_(-initrange, initrange)
        self.vid_linear_v2.bias.data.fill_(0)

        self.sen_linear_v2.weight.data.uniform_(-initrange, initrange)
        self.sen_linear_v2.bias.data.fill_(0)

        self.att_linear_v2.weight.data.uniform_(-initrange, initrange)
        self.att_linear_v2.bias.data.fill_(0)

        self.h2o.weight.data.uniform_(-initrange, initrange)
        self.h2o.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state

    def init_hidden_new(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state



    # video_fc_feats: batch * encode_time_step * fc_feat_size
    def forward(self, video_fc_feat, video_caption, cap_length_list=None):
        #pdb.set_trace()
        batch_size = video_fc_feat.size(0)
        batch_size_caption = video_caption.size(0)

        video_state = self.init_hidden_new(batch_size)
        caption_state = self.init_hidden_new(batch_size_caption)

        # 将 caption 用 LSTM 进行编码, 用来 soft attention
        caption_outputs = []
        caption_time_step = video_caption.size(1)
        for i in range(caption_time_step):
            word = video_caption[:, i].clone()
            if video_caption[:, i].data.sum() == 0:
                break
            #import ipdb
            #pdb.set_trace()
            caption_xt = self.word_embedding(word)
            #caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_outputs.append(caption_output)
            caption_state = (caption_output, caption_state)
        # caption_outputs: batch * caption_time_step * lstm_hidden_size
        caption_outputs = torch.cat([_.unsqueeze(1) for _ in caption_outputs], 1).contiguous()

        # 将 video fc feat 用 LSTM 进行编码
        video_outputs = []
        for i in range(self.video_time_step):
            video_xt = self.video_embedding(video_fc_feat[:, i, :])
            video_output, video_state = self.lstm_video.forward(video_xt, video_state)
            video_outputs.append(video_output)
            video_state = (video_output, video_state)
        # video_outputs: batch * video_time_step * lstm_hidden_size
        video_outputs = torch.cat([_.unsqueeze(1) for _ in video_outputs], 1).contiguous()

        # soft attention for caption based on each video
        output_list = list()
        for i in range(self.video_time_step):
            # part 1
            video_outputs_linear = self.vid_linear(video_outputs[:, i, :])
            video_outputs_linear_expand = video_outputs_linear.expand(caption_outputs.size(1), video_outputs_linear.size(0),
                                                                      video_outputs_linear.size(1)).transpose(0, 1)

            # part 2
            caption_outputs_flatten = caption_outputs.view(-1, self.lstm_hidden_size)
            caption_outputs_linear = self.sen_linear(caption_outputs_flatten)
            caption_outputs_linear = caption_outputs_linear.view(batch_size_caption, caption_outputs.size(1), self.att_hidden_size)

            # part 1 and part 2 attention
            sig_probs = []
            for cap_id in range(batch_size_caption):
                cap_length = max(cap_length_list[cap_id], 1)
                caption_output_linear_cap_id = caption_outputs_linear[cap_id, : cap_length, :]
                video_outputs_linear_expand_clip = video_outputs_linear_expand[:, :cap_length, :]
                caption_outputs_linear_cap_id_exp = caption_output_linear_cap_id.expand_as(video_outputs_linear_expand_clip) 
                video_caption = F.tanh(video_outputs_linear_expand_clip \
                        + caption_outputs_linear_cap_id_exp)
                
                #video_caption = video_outputs_linear_expand_clip \
                #        + caption_outputs_linear_cap_id_exp
                
                #pdb.set_trace()
                video_caption_view = video_caption.contiguous().view(-1, self.att_hidden_size)
                video_caption_out = self.att_linear(video_caption_view)
                video_caption_out_view = video_caption_out.view(-1, cap_length)
                atten_weights = nn.Softmax(dim=1)(video_caption_out_view).unsqueeze(2)

                # http://pytorch.org/docs/master/torch.html#torch.bmm
                # batch 1: b x n x m, batch 2: b x m x p, 结果为: b x n x p
                caption_output_cap_id = caption_outputs[cap_id, : cap_length, :]
                caption_output_cap_id_exp = caption_output_cap_id.expand(batch_size,\
                        caption_output_cap_id.size(0), caption_output_cap_id.size(1))
                atten_caption = torch.bmm(caption_output_cap_id_exp.transpose(1, 2), atten_weights).squeeze(2)

                video_caption_hidden = torch.cat((atten_caption, video_outputs[:, i, :]), dim=1)
                #cur_probs = self.h2o(video_caption_hidden)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #cur_probs = F.sigmoid(self.h2o(video_caption_hidden))
                cur_probs = cos(atten_caption, video_outputs[: ,i, :]).unsqueeze(1)
                #pdb.set_trace()
                #cur_probs = torch.sigmoid(cur_probs)


                sig_probs.append(cur_probs)

            sig_probs = torch.cat([_ for _ in sig_probs], 1).contiguous()
            output_list.append(sig_probs)
        simMM = torch.stack(output_list, dim=2).mean(2)
       
        output_list_v2 = list()
        for i in range(caption_time_step):
            # part 1
            caption_outputs_linear = self.sen_linear_v2(caption_outputs[:, i, :])
            caption_outputs_linear_expand = caption_outputs_linear.expand(self.video_time_step, caption_outputs_linear.size(0),
                                                                      caption_outputs_linear.size(1)).transpose(0, 1)
            # part 2
            video_outputs_flatten = video_outputs.view(-1, self.lstm_hidden_size)
            video_outputs_linear = self.vid_linear_v2(video_outputs_flatten)
            video_outputs_linear = video_outputs_linear.view(batch_size, self.video_time_step, self.att_hidden_size)

            # part 1 and part 2 attention
            sig_probs = []
            for vid_id in range(batch_size):
                video_outputs_linear_video_id = video_outputs_linear[vid_id, : , :]
                video_outputs_linear_video_id_exp = video_outputs_linear_video_id.expand_as(caption_outputs_linear_expand) 
                video_caption = F.tanh(caption_outputs_linear_expand \
                        + video_outputs_linear_video_id_exp)
                
                video_caption_view = video_caption.contiguous().view(-1, self.att_hidden_size)
                video_caption_out = self.att_linear_v2(video_caption_view)
                video_caption_out_view = video_caption_out.view(-1, self.video_time_step)
                atten_weights = nn.Softmax(dim=1)(video_caption_out_view).unsqueeze(2)

                video_output_video_id = video_outputs[vid_id]
                video_output_video_id_exp = video_output_video_id.expand(batch_size_caption,\
                        video_output_video_id.size(0), video_output_video_id.size(1))
                atten_video = torch.bmm(video_output_video_id_exp.transpose(1, 2), atten_weights).squeeze(2)

                video_caption_hidden = torch.cat((atten_video, caption_outputs[:, i, :]), dim=1)
                #cur_probs = self.h2o(video_caption_hidden)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #cur_probs = F.sigmoid(self.h2o(video_caption_hidden))
                cur_probs = cos(atten_video, caption_outputs[: ,i, :]).unsqueeze(1)
                #pdb.set_trace()
                #cur_probs = torch.sigmoid(cur_probs)


                sig_probs.append(cur_probs)
            sig_probs = torch.cat([_ for _ in sig_probs], 1).contiguous()
            output_list_v2.append(sig_probs)
        
        simMM_v2 = torch.stack(output_list_v2, dim=2).mean(2)
        print(torch.max(simMM_v2))
        print(torch.min(simMM_v2))

        return simMM + simMM_v2.transpose(0, 1)


class SSTV3(nn.Module):
    def __init__(self, opt):
        super(SSTV3, self).__init__()

        self.word_cnt = opt.word_cnt
        self.fc_feat_size = opt.fc_feat_size
        self.video_embedding_size = opt.video_embedding_size
        self.word_embedding_size = opt.word_embedding_size
        self.lstm_hidden_size = opt.lstm_hidden_size
        self.video_time_step = opt.video_time_step
        self.caption_time_step = opt.caption_time_step
        self.dropout_prob = opt.dropout_prob
        self.n_anchors = opt.n_anchors
        self.att_hidden_size = opt.att_hidden_size

        self.video_embedding = nn.Linear(self.fc_feat_size, self.video_embedding_size)

        #self.lstm_video = LSTMCore(self.video_embedding_size, self.lstm_hidden_size, self.dropout_prob)
        #self.lstm_caption = LSTMCore(self.word_embedding_size, self.lstm_hidden_size, self.dropout_prob)
        self.lstm_caption = torch.nn.LSTMCell(self.word_embedding_size, self.lstm_hidden_size, bias=True)
        self.vid_linear = nn.Linear(self.video_embedding_size, self.att_hidden_size)
        self.sen_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        # nn.Tanh 是一种不带任何参数的 operation，所以直接在 forward 里面调用 F.tanh 即可
        # self.tanh_att = nn.Tanh()

        self.att_linear = nn.Linear(self.att_hidden_size, 1)

        self.h2o = nn.Linear(self.lstm_hidden_size * 2, self.n_anchors)

        # self.softmax_att = nn.Softmax()

        # 注意因为 idx_to_word 是从 1 开始, 此处要加 1, 要不然会遇到 bug:
        # cuda runtime error (59): device-side assert triggered
        #self.word_embedding = nn.Embedding(self.word_cnt + 1, self.word_embedding_size)
        self.word_embedding = nn.Linear(300, self.word_embedding_size)
        # self.vocab_logit = nn.Linear(self.lstm_hidden_size, self.word_cnt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.video_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_embedding.bias.data.fill_(0)

        self.word_embedding.weight.data.uniform_(-initrange, initrange)
        self.word_embedding.bias.data.fill_(0)

        # self.vocab_logit.weight.data.uniform_(-initrange, initrange)
        # self.vocab_logit.bias.data.fill_(0)

        self.vid_linear.weight.data.uniform_(-initrange, initrange)
        self.vid_linear.bias.data.fill_(0)

        self.sen_linear.weight.data.uniform_(-initrange, initrange)
        self.sen_linear.bias.data.fill_(0)

        self.att_linear.weight.data.uniform_(-initrange, initrange)
        self.att_linear.bias.data.fill_(0)

        self.h2o.weight.data.uniform_(-initrange, initrange)
        self.h2o.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state

    def init_hidden_new(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state



    # video_fc_feats: batch * encode_time_step * fc_feat_size
    def forward(self, video_fc_feat, video_caption, cap_length_list=None):
        #pdb.set_trace()
        batch_size = video_fc_feat.size(0)
        batch_size_caption = video_caption.size(0)

        video_state = self.init_hidden_new(batch_size)
        caption_state = self.init_hidden_new(batch_size_caption)

        # 将 caption 用 LSTM 进行编码, 用来 soft attention
        caption_outputs = []
        caption_time_step = video_caption.size(1)
        for i in range(caption_time_step):
            word = video_caption[:, i].clone()
            if video_caption[:, i].data.sum() == 0:
                break
            #import ipdb
            #pdb.set_trace()
            caption_xt = self.word_embedding(word)
            #caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_outputs.append(caption_output)
            caption_state = (caption_output, caption_state)
        # caption_outputs: batch * caption_time_step * lstm_hidden_size
        caption_outputs = torch.cat([_.unsqueeze(1) for _ in caption_outputs], 1).contiguous()

        #pdb.set_trace()
        # video_fc_output: batch * video_embedding_size
        video_fc_output = self.video_embedding(video_fc_feat.mean(1))
        #video_fc_output = nn.functional.relu(video_fc_output)
        # part 1
        # video_output_linear: batch_size* self.att_hidden_size
        video_output_linear = self.vid_linear(video_fc_output)
        #video_output_linear = nn.functional.relu(video_output_linear)
        # video_outputs_linear_expand: batch_size* self.att_hidden_size
        video_outputs_linear_expand = video_output_linear.expand(caption_outputs.size(1), video_output_linear.size(0), \
                video_output_linear.size(1)).transpose(0, 1)

        # part 2
        caption_outputs_flatten = caption_outputs.view(-1, self.lstm_hidden_size)
        caption_outputs_linear = self.sen_linear(caption_outputs_flatten)
        caption_outputs_linear = caption_outputs_linear.view(batch_size_caption, caption_outputs.size(1), self.att_hidden_size)

        # part 1 and part 2 attention
        sig_probs = []
        for cap_id in range(batch_size_caption):
            cap_length = max(cap_length_list[cap_id], 1)
            caption_output_linear_cap_id = caption_outputs_linear[cap_id, : cap_length, :]
            video_outputs_linear_expand_clip = video_outputs_linear_expand[:, :cap_length, :]
            caption_outputs_linear_cap_id_exp = caption_output_linear_cap_id.expand_as(video_outputs_linear_expand_clip) 
            video_caption = F.tanh(video_outputs_linear_expand_clip \
                    + caption_outputs_linear_cap_id_exp)
            
            #pdb.set_trace()
            video_caption_view = video_caption.contiguous().view(-1, self.att_hidden_size)
            video_caption_out = self.att_linear(video_caption_view)
            video_caption_out_view = video_caption_out.view(-1, cap_length)
            atten_weights = nn.Softmax(dim=1)(video_caption_out_view).unsqueeze(2)

            # http://pytorch.org/docs/master/torch.html#torch.bmm
            # batch 1: b x n x m, batch 2: b x m x p, 结果为: b x n x p
            caption_output_cap_id = caption_outputs[cap_id, : cap_length, :]
            caption_output_cap_id_exp = caption_output_cap_id.expand(batch_size,\
                    caption_output_cap_id.size(0), caption_output_cap_id.size(1))
            atten_caption = torch.bmm(caption_output_cap_id_exp.transpose(1, 2), atten_weights).squeeze(2)

            video_caption_hidden = torch.cat((atten_caption, video_fc_output), dim=1)
            #cur_probs = self.h2o(video_caption_hidden)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        #cur_probs = F.sigmoid(self.h2o(video_caption_hidden))
            cur_probs = cos(atten_caption, video_fc_output).unsqueeze(1)
            #pdb.set_trace()
            #cur_probs = torch.sigmoid(cur_probs)
            sig_probs.append(cur_probs)

        simMM = torch.cat([_ for _ in sig_probs], 1).contiguous()
    
        print(torch.max(simMM))
        print(torch.min(simMM))
        return simMM


class SSTV4(nn.Module):
    def __init__(self, opt):
        super(SSTV4, self).__init__()

        self.word_cnt = opt.word_cnt
        self.fc_feat_size = opt.fc_feat_size
        self.video_embedding_size = opt.video_embedding_size
        self.word_embedding_size = opt.word_embedding_size
        self.lstm_hidden_size = opt.lstm_hidden_size
        self.video_time_step = opt.video_time_step
        self.caption_time_step = opt.caption_time_step
        self.dropout_prob = opt.dropout_prob
        self.n_anchors = opt.n_anchors
        self.att_hidden_size = opt.att_hidden_size

        self.video_embedding = nn.Linear(self.fc_feat_size, self.video_embedding_size)
        self.video_embedding_mean = nn.Linear(self.fc_feat_size, self.video_embedding_size)

        self.lstm_caption = torch.nn.LSTMCell(self.word_embedding_size, self.lstm_hidden_size, bias=True)
        self.lstm_video = torch.nn.LSTMCell(self.video_embedding_size, self.lstm_hidden_size, bias=True)
        self.vid_linear_st1 = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        self.sen_linear_st1 = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        self.vid_linear_st2 = nn.Linear(self.video_embedding_size, self.att_hidden_size)
        self.sen_linear_st2 = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        # nn.Tanh 是一种不带任何参数的 operation，所以直接在 forward 里面调用 F.tanh 即可
        # self.tanh_att = nn.Tanh()

        self.att_linear_st1 = nn.Linear(self.att_hidden_size, 1)
        self.att_linear_st2 = nn.Linear(self.att_hidden_size, 1)

        self.h2o = nn.Linear(self.lstm_hidden_size * 2, self.n_anchors)

        self.word_embedding = nn.Linear(300, self.word_embedding_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.video_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_embedding.bias.data.fill_(0)

        self.word_embedding.weight.data.uniform_(-initrange, initrange)
        self.word_embedding.bias.data.fill_(0)

        # self.vocab_logit.weight.data.uniform_(-initrange, initrange)
        # self.vocab_logit.bias.data.fill_(0)

        self.vid_linear_st1.weight.data.uniform_(-initrange, initrange)
        self.vid_linear_st1.bias.data.fill_(0)
        self.vid_linear_st2.weight.data.uniform_(-initrange, initrange)
        self.vid_linear_st2.bias.data.fill_(0)

        self.sen_linear_st1.weight.data.uniform_(-initrange, initrange)
        self.sen_linear_st2.weight.data.uniform_(-initrange, initrange)
        self.sen_linear_st1.bias.data.fill_(0)
        self.sen_linear_st2.bias.data.fill_(0)

        self.att_linear_st1.weight.data.uniform_(-initrange, initrange)
        self.att_linear_st2.weight.data.uniform_(-initrange, initrange)
        self.att_linear_st1.bias.data.fill_(0)
        self.att_linear_st2.bias.data.fill_(0)

        self.h2o.weight.data.uniform_(-initrange, initrange)
        self.h2o.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(1, batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state

    def init_hidden_new(self, batch_size):
        weight = next(self.parameters()).data
        init_h = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_c = Variable(weight.new(batch_size, self.lstm_hidden_size).zero_())
        init_state = (init_h, init_c)

        return init_state


    # video_fc_feats: batch * encode_time_step * fc_feat_size
    def forward(self, video_fc_feat, video_caption, cap_length_list=None):
        #pdb.set_trace()
        batch_size = video_fc_feat.size(0)
        batch_size_caption = video_caption.size(0)

        video_state = self.init_hidden_new(batch_size)
        caption_state = self.init_hidden_new(batch_size_caption)

        # 将 caption 用 LSTM 进行编码, 用来 soft attention
        caption_outputs = []
        caption_time_step = video_caption.size(1)
        for i in range(caption_time_step):
            word = video_caption[:, i].clone()
            if video_caption[:, i].data.sum() == 0:
                break
            #import ipdb
            #pdb.set_trace()
            caption_xt = self.word_embedding(word)
            #caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_outputs.append(caption_output)
            caption_state = (caption_output, caption_state)
        # caption_outputs: batch * caption_time_step * lstm_hidden_size
        caption_outputs = torch.cat([_.unsqueeze(1) for _ in caption_outputs], 1).contiguous()


        # video_fc_output: batch * video_embedding_size
        video_outputs = []
        for i in range(self.video_time_step):
            video_xt = self.video_embedding(video_fc_feat[:, i, :])
            video_output, video_state = self.lstm_video.forward(video_xt, video_state)
            video_outputs.append(video_output)
            video_state = (video_output, video_state)
        #pdb.set_trace()
        video_outputs = torch.cat([_.unsqueeze(1) for _ in video_outputs], 1).contiguous()
        # video_seq_outputs_linear: batch * video_time_step * self.att_hidden_size
        video_seq_outputs_linear = self.vid_linear_st2(video_outputs)
        #video_seq_outputs_linear = nn.functional.relu(video_seq_outputs_linear)
        
        # stage 1, part 1
        video_fc_output = self.video_embedding_mean(video_fc_feat.mean(1))
        video_fc_output = nn.functional.relu(video_fc_output)
        video_output_linear = self.vid_linear_st1(video_fc_output)
        #video_output_linear = nn.functional.relu(video_output_linear)
        # video_outputs_linear_expand: batch_size* self.att_hidden_size
        video_outputs_linear_expand = video_output_linear.expand(caption_outputs.size(1), video_output_linear.size(0), \
                video_output_linear.size(1)).transpose(0, 1)

        # stage 1, part 2
        caption_outputs_flatten = caption_outputs.view(-1, self.lstm_hidden_size)
        caption_outputs_linear = self.sen_linear_st1(caption_outputs_flatten)
        caption_outputs_linear = caption_outputs_linear.view(batch_size_caption, caption_outputs.size(1), self.att_hidden_size)

        # stage 1, part 1 and part 2 attention
        sig_probs = []
        for cap_id in range(batch_size_caption):
            cap_length = max(cap_length_list[cap_id], 1)
            caption_output_linear_cap_id = caption_outputs_linear[cap_id, : cap_length, :]
            video_outputs_linear_expand_clip = video_outputs_linear_expand[:, :cap_length, :]
            caption_outputs_linear_cap_id_exp = caption_output_linear_cap_id.expand_as(video_outputs_linear_expand_clip) 
            video_caption = F.tanh(video_outputs_linear_expand_clip \
                    + caption_outputs_linear_cap_id_exp)
            
            #pdb.set_trace()
            video_caption_view = video_caption.contiguous().view(-1, self.att_hidden_size)
            video_caption_out = self.att_linear_st1(video_caption_view)
            video_caption_out_view = video_caption_out.view(-1, cap_length)
            atten_weights = nn.Softmax(dim=1)(video_caption_out_view).unsqueeze(2)

            caption_output_cap_id = caption_outputs[cap_id, : cap_length, :]
            caption_output_cap_id_exp = caption_output_cap_id.expand(batch_size,\
                    caption_output_cap_id.size(0), caption_output_cap_id.size(1))
            

            # atten_caption: batch_size*rp_num, self.lstm_hidden_size 
            atten_caption = torch.bmm(caption_output_cap_id_exp.transpose(1, 2), atten_weights).squeeze(2)

            #att_ftr_list.append(atten_caption)
            # atten_caption_exp: batch * video_time_step * self.att_hidden_size
            atten_caption_exp = atten_caption.expand(self.video_time_step, atten_caption.size(0), atten_caption.size(1)).transpose(0, 1) 
            video_caption_st2 = F.tanh(video_seq_outputs_linear \
                    + atten_caption_exp)
            video_caption_out_st2 = self.att_linear_st2(video_caption_st2)
            video_caption_out_st2_view = video_caption_out_st2.view(-1, self.video_time_step)
            atten_weights_st2 = nn.Softmax(dim=1)(video_caption_out_st2_view).unsqueeze(2)
            # atten_video: batch * rp_num, self.att_hidden_size
            atten_video = torch.bmm(video_seq_outputs_linear.transpose(1, 2), atten_weights_st2).squeeze(2)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cur_probs = cos(atten_video, atten_caption).unsqueeze(1)
            sig_probs.append(cur_probs)

        simMM = torch.cat([_ for _ in sig_probs], 1).contiguous()
        print(torch.max(simMM))
        print(torch.min(simMM))
        return simMM


