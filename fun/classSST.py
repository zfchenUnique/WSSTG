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
        self.att_hidden_size = opt.att_hidden_size

        self.video_embedding = nn.Linear(self.fc_feat_size, self.video_embedding_size)

        self.lstm_video = torch.nn.LSTMCell(self.video_embedding_size, self.lstm_hidden_size, bias=True)
        self.lstm_caption = torch.nn.LSTMCell(self.word_embedding_size, self.lstm_hidden_size, bias=True)


        self.vid_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)
        self.sen_linear = nn.Linear(self.lstm_hidden_size, self.att_hidden_size)

        self.att_linear = nn.Linear(self.att_hidden_size, 1)

        self.word_embedding = nn.Linear(300, self.word_embedding_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.video_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_embedding.bias.data.fill_(0)

        self.word_embedding.weight.data.uniform_(-initrange, initrange)
        self.word_embedding.bias.data.fill_(0)

        self.vid_linear.weight.data.uniform_(-initrange, initrange)
        self.vid_linear.bias.data.fill_(0)

        self.sen_linear.weight.data.uniform_(-initrange, initrange)
        self.sen_linear.bias.data.fill_(0)

        self.att_linear.weight.data.uniform_(-initrange, initrange)
        self.att_linear.bias.data.fill_(0)


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

    def forward(self, video_fc_feat, video_caption, cap_length_list=None):
        if self.training:
            return self.forward_training(video_fc_feat, video_caption, cap_length_list)
        else:
            return self.forward_val(video_fc_feat, video_caption, cap_length_list)

    # video_fc_feats: batch * encode_time_step * fc_feat_size
    def forward_training(self, video_fc_feat, video_caption, cap_length_list=None):
        #pdb.set_trace()
        batch_size = video_fc_feat.size(0)
        batch_size_caption = video_caption.size(0)

        video_state = self.init_hidden_new(batch_size)
        caption_state = self.init_hidden_new(batch_size_caption)

        caption_outputs = []
        caption_time_step = video_caption.size(1)
        for i in range(caption_time_step):
            word = video_caption[:, i].clone()
            #if video_caption[:, i].data.sum() == 0:
            #    break
            #import ipdb
            #pdb.set_trace()
            caption_xt = self.word_embedding(word)
            #caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_outputs.append(caption_output)
            caption_state = (caption_output, caption_state)
        # caption_outputs: batch * caption_time_step * lstm_hidden_size
        caption_outputs = torch.cat([_.unsqueeze(1) for _ in caption_outputs], 1).contiguous()

        # LSTM encoding 
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
                        + caption_outputs_linear_cap_id_exp)
                
                video_caption_view = video_caption.contiguous().view(-1, self.att_hidden_size)
                video_caption_out = self.att_linear(video_caption_view)
                video_caption_out_view = video_caption_out.view(-1, cap_length)
                atten_weights = nn.Softmax(dim=1)(video_caption_out_view).unsqueeze(2)

                caption_output_cap_id = caption_outputs[cap_id, : cap_length, :]
                caption_output_cap_id_exp = caption_output_cap_id.expand(batch_size,\
                        caption_output_cap_id.size(0), caption_output_cap_id.size(1))
                atten_caption = torch.bmm(caption_output_cap_id_exp.transpose(1, 2), atten_weights).squeeze(2)

                video_caption_hidden = torch.cat((atten_caption, video_outputs[:, i, :]), dim=1)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                cur_probs = cos(atten_caption, video_outputs[: ,i, :]).unsqueeze(1)


                sig_probs.append(cur_probs)

            sig_probs = torch.cat([_ for _ in sig_probs], 1).contiguous()
            output_list.append(sig_probs)
        simMM = torch.stack(output_list, dim=2).mean(2)
        return simMM

    # video_fc_feats: batch * encode_time_step * fc_feat_size
    def forward_val(self, video_fc_feat, video_caption, cap_length_list=None):
        #pdb.set_trace()
        batch_size = video_fc_feat.size(0)
        batch_size_caption = video_caption.size(0)
        num_tube_per_video = int(batch_size / batch_size_caption) 

        video_state = self.init_hidden_new(batch_size)
        caption_state = self.init_hidden_new(batch_size_caption)

        caption_outputs = []
        caption_time_step = video_caption.size(1)
        for i in range(caption_time_step):
            word = video_caption[:, i].clone()
            #if video_caption[:, i].data.sum() == 0:
            #    break
            #import ipdb
            #pdb.set_trace()
            caption_xt = self.word_embedding(word)
            #caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_output, caption_state = self.lstm_caption.forward(caption_xt, caption_state)
            caption_outputs.append(caption_output)
            caption_state = (caption_output, caption_state)
        # caption_outputs: batch * caption_time_step * lstm_hidden_size
        caption_outputs = torch.cat([_.unsqueeze(1) for _ in caption_outputs], 1).contiguous()

        # LSTM encoding 
        video_outputs = []
        for i in range(self.video_time_step):
            video_xt = self.video_embedding(video_fc_feat[:, i, :])
            video_output, video_state = self.lstm_video.forward(video_xt, video_state)
            video_outputs.append(video_output)
            video_state = (video_output, video_state)
        # video_outputs: batch * video_time_step * lstm_hidden_size
        video_outputs = torch.cat([_.unsqueeze(1) for _ in video_outputs], 1).contiguous()

        # batch_size_caption * num_word * lstm_hidden_size  
        caption_outputs_linear = self.sen_linear(caption_outputs)
       
       # soft attention for caption based on each video
        output_list = list()
        for i in range(self.video_time_step):
            # batch_size * lstm_hidden_size   
            video_outputs_linear = self.vid_linear(video_outputs[:, i, :])
            # batch_size * num_word * lstm_hidden_size   
            video_outputs_linear_expand = video_outputs_linear.expand(caption_outputs.size(1), video_outputs_linear.size(0),
                                                                      video_outputs_linear.size(1)).transpose(0, 1)

            # part 1 and part 2 attention
            sig_probs = []
            
            for cap_id in range(batch_size_caption):
                
                tube_id_st = cap_id*num_tube_per_video 
                tube_id_ed = (cap_id+1)*num_tube_per_video 

                cap_length = max(cap_length_list[cap_id], 1)
                # num_tube_per_video * cap_length * lstm_hidden_size 
                tube_outputs_aligned = video_outputs_linear_expand[tube_id_st:tube_id_ed, : cap_length, :] 
                # cap_length * lstm_hidden_size 
                caption_output_linear_cap_id = caption_outputs_linear[cap_id, : cap_length, :]
                # num_tube_per_video * cap_length * lstm_hidden_size 
                caption_outputs_linear_cap_id_exp = caption_output_linear_cap_id.expand_as(tube_outputs_aligned) 
                # num_tube_per_video * cap_length * lstm_hidden_size 
                video_caption = F.tanh(tube_outputs_aligned  \
                        + caption_outputs_linear_cap_id_exp)
                
                video_caption_view = video_caption.contiguous().view(-1, self.att_hidden_size)
                video_caption_out = self.att_linear(video_caption_view)
                video_caption_out_view = video_caption_out.view(-1, cap_length)
                
                # num_tube_per_video * cap_length
                atten_weights = nn.Softmax(dim=1)(video_caption_out_view).unsqueeze(2)

                # cap_length * lstm_hidden_size 
                caption_output_cap_id = caption_outputs[cap_id, : cap_length, :]
                # num_tube_per_video * cap_length * lstm_hidden_size 
                caption_output_cap_id_exp = caption_output_cap_id.expand(num_tube_per_video,\
                        caption_output_cap_id.size(0), caption_output_cap_id.size(1))
                # num_tube_per_video * lstm_hidden_size 
                atten_caption = torch.bmm(caption_output_cap_id_exp.transpose(1, 2), atten_weights).squeeze(2)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                # num_tube_per_video
                cur_probs = cos(atten_caption, video_outputs[tube_id_st:tube_id_ed, i, :]).unsqueeze(1)
                # num_tube_per_video
                sig_probs.append(cur_probs)

            # num_tube_per_video * batch_size_caption 
            sig_probs = torch.cat([_ for _ in sig_probs], 1).contiguous()
            output_list.append(sig_probs)
        # num_tube_per_video * batch_size_caption 
        simMM = torch.stack(output_list, dim=2).mean(2)
        # batch_size_caption * num_tube_per_video  
        simMM = simMM.transpose(0, 1)
        #pdb.set_trace()
        return simMM

