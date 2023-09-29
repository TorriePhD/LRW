from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class LmkRNN(nn.Module):

    def __init__(self, feature_size=40, hidden_size=1024, out_size=500, n_layer=3, bf=False, bi=True):
        super(LmkRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer

        self.batch_size = 0
        self.max_seq_len = 0

        self.size_bl_out = feature_size * 2

        # n_direction = 2 if bi else 1
        # self.n_direction = n_direction
        self.n_direction = 1

        self.rnn_en = nn.GRU(input_size=self.size_bl_out, hidden_size=hidden_size,
                             num_layers=n_layer, batch_first=bf, bidirectional=bi)
        self.h_en = 0

        self.fc1 = nn.Linear(hidden_size, out_size)

    def encoder(self, seq, seq_len):
        # seq: (max_seq_len, batch_size, n_pts, 2)
        max_seq_len, batch_size, n_pts = seq.shape
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        seq_c = torch.tanh(seq)
        # seq_c: maxSeqLen, batchSize, self.size_bl_out

        packed = pack_padded_sequence(
            seq_c, seq_len, batch_first=False, enforce_sorted=False)

        out, h_n = self.rnn_en(packed)
        self.h_en = h_n

    def forward(self, seq, seq_len):
        self.encoder(seq, seq_len)

        enc = self.h_en[-1]
        out = self.fc1(enc)
        # out = torch.sigmoid(out)

        return F.normalize(out)

    def get_embedding(self, seq, seq_len, device):
        out = self.forward(seq=seq, seq_len=seq_len)
        return out.detach().cpu().numpy()
