from .video_cnn import VideoCNN
import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast, GradScaler


class VideoModel(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()   
        
        self.args = args
        
        # self.video_cnn = VideoCNN(se=self.args.se)  
        self.video_fn = nn.Linear(40*2, 512)      #number of landmarks * 2 (x and y) = 40
        if(self.args.border):
            in_dim = 512 + 1
        else:
            in_dim = 512
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)        
            

        self.v_cls = nn.Linear(1024, self.args.n_class)     
        self.dropout = nn.Dropout(p=dropout)        

    def forward(self, v,seq_len, border=None):
        # self.gru.flatten_parameters()
        
        if(self.training):                            
            with autocast():
                
                f_v = self.video_fn(v)  
                f_v = self.dropout(f_v)        
            f_v = f_v.float()
        else:                            
            f_v = self.video_fn(v)  
            f_v = self.dropout(f_v)        
        
        if(self.args.border):
            border = border[:,:,None]
            h, _ = self.gru(torch.cat([f_v, border], -1))
        else:            
            packed = pack_padded_sequence(
            f_v, seq_len, batch_first=False, enforce_sorted=False)
            _, h = self.gru(packed)
                                                                                                        
        y_v = self.v_cls(self.dropout(h)).mean(0)
            
        return y_v