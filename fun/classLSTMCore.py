from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

import opts


class LSTMCore(nn.Module):
    def __init__(self, input_encoding_size, lstm_size, dropout_prob):
        super(LSTMCore, self).__init__()

        self.input_encoding_size = input_encoding_size
        self.lstm_size = lstm_size
        self.dropout_prob = dropout_prob

        # build a LSTM
        self.i2h = nn.Linear(self.input_encoding_size, 4 * self.lstm_size)
        self.h2h = nn.Linear(self.lstm_size, 4 * self.lstm_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        # init
        initrange = 0.1
        self.i2h.weight.data.uniform_(-initrange, initrange)
        self.i2h.bias.data.uniform_(-initrange, initrange)

        self.h2h.weight.data.uniform_(-initrange, initrange)
        self.h2h.bias.data.uniform_(-initrange, initrange)

    # state = (pre_h, pre_c)
    def forward(self, xt, state):
        pre_h = state[0][-1]
        pre_c = state[1][-1]

        all_input_sums = self.i2h(xt) + self.h2h(pre_h)

        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.lstm_size)
        sigmoid_chunk_sig = F.sigmoid(sigmoid_chunk)

        in_gate = sigmoid_chunk_sig.narrow(1, 0, self.lstm_size)
        forget_gate = sigmoid_chunk_sig.narrow(1, self.lstm_size, self.lstm_size)
        out_gate = sigmoid_chunk_sig.narrow(1, self.lstm_size * 2, self.lstm_size)

        in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.lstm_size, self.lstm_size))

        next_c = forget_gate * pre_c + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)
        next_h = self.dropout(next_h)

        output = next_h
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))

        return output, state
