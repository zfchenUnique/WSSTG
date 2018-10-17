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
        print(torch.max(simMM))
        print(torch.min(simMM))
        #sig_probs = torch.cat([_.unsqueeze(1) for _ in sig_probs], 1).contiguous()
        #sig_probs = nn.Softmax()(sig_probs.view(batch_size, -1)).view(batch_size,video_outputs.size(1),-1)
        return simMM
