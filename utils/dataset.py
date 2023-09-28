# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from .cvtransforms import *
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence


# jpeg = TurboJPEG()
class LRWDataset(Dataset):
    def __init__(self, phase, args):

        with open('/home/st392/code/sandbox/learn-an-effective-lip-reading-model-without-pains/label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        
        self.list = []
        self.unlabel_list = []
        self.phase = phase        
        self.args = args
        
        # if(not hasattr(self.args, 'is_aug')):
        #     setattr(self.args, 'is_aug', True)
        parentPath = Path("/home/st392/code/datasets/LRW/facemeshes")
        for (i, label) in enumerate(self.labels):
            myPath = parentPath / self.phase / f"{label}_meshes.npy"
            meshes = np.load(myPath, allow_pickle=True)
            #torch.flatten(seq, start_dim=-2)
            # self.list += [[torch.from_numpy(file),i] for file in meshes]
            self.list += [[torch.flatten(torch.from_numpy(self.normalize(file)), start_dim=-2),i] for file in meshes]
            
    def normalize(self, lmks):
            median = np.median(lmks, axis=0)
            lmks = lmks - median
            # lmks /= 10
            return lmks

        
    def __getitem__(self, idx):
        inputs,label = self.list[idx]   
        return inputs, label

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])
        
        tensor = torch.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor            
    def custom_collate(self,batch):
        # Sort batch by sequence length in descending order
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Extract sequences and labels from the batch
        sequences, labels = zip(*batch)
        sequences = [seq.float() for seq in sequences]
        # Get sequence lengths and create a tensor
        seq_lengths = [len(seq) for seq in sequences]
        
        # Pad the sequences
        sequences_padded = pad_sequence(sequences, batch_first=False, padding_value=0)
        
        # Convert labels to tensor
        labels = torch.LongTensor(labels)
        return sequences_padded, seq_lengths, labels
if __name__ == "__main__":
    LRWDataset('train', None)
    data,label = LRWDataset[0]
    print(data.shape)