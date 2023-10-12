# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from torchvision.io import read_video
try:
    from .cvtransforms import *
except:
    from cvtransforms import *
import torch
from pathlib import Path

# jpeg = TurboJPEG()
class LRWDataset(Dataset):
    def __init__(self, phase, args):

        with open('/home/st392/code/learn-an-effective-lip-reading-model-without-pains/label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        self.labels = sorted(self.labels)
        self.list = []
        self.unlabel_list = []
        self.phase = phase        
        self.args = args
        
        # if(not hasattr(self.args, 'is_aug')):
        #     setattr(self.args, 'is_aug', True)

            
        self.list = sorted(list(Path("/home/st392/code/datasets/LRW/lipCrop_mp4").rglob(f'{phase}/*.mp4')))
        # self.durations = [self.load_duration(str(file).replace('lipCrop_mp4', 'lipread_mp4').replace('.mp4', '.txt')) for file in self.list]
        
    def __getitem__(self, idx):
            
        video_file = self.list[idx]
        video, _, _ = read_video(str(video_file), pts_unit='sec')  # Read video using torchvision.io.read_video

        # Assuming the videos are grayscale, but if they are RGB, you may need to convert them to grayscale
        # using some additional processing.
        video = video.float() / 255.0  # Normalize the video tensor
        video = video[:,:,:,0]
        video = video.unsqueeze(1)
        if(self.phase == 'train'):
            batch_img = TensorRandomCrop(video, (88, 88))
            batch_img = TensorRandomFlip(batch_img)
        elif self.phase == 'val' or self.phase == 'test':
            batch_img = CenterCrop(video, (88, 88))
        
        result = {}            
        result['video'] = batch_img
        #print(result['video'].size())
        #get index in self.label where the label is the same as the videofle
        result['label'] = torch.tensor(self.labels.index(video_file.parent.parent.name))

        return result

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
def LRW_collate(batch):
    batch_size = len(batch)
    max_len = max([batch[i]['video'].size(0) for i in range(batch_size)])
    max_w = max([batch[i]['video'].size(2) for i in range(batch_size)])
    max_h = max([batch[i]['video'].size(3) for i in range(batch_size)])
    padded_videos = torch.zeros(batch_size, max_len, 1,max_w, max_h)
    labels = torch.zeros(batch_size, 500, dtype=torch.float32)
    for i in range(batch_size):
        video = batch[i]['video']
        label = batch[i]['label']
        padded_videos[i, :video.size(0), :,:video.size(2), :video.size(3)] = video
        #convert label to one hot encoding float tensor
        labels[i, :] = torch.nn.functional.one_hot(label, 500).float()
    return {'video': padded_videos, 'label': labels}

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = LRWDataset('val', None)
    batch_size = 10
    num_workers = 0
    shuffle = False
    collate_fn = LRW_collate
    loader =  DataLoader(dataset,
            batch_size = batch_size, 
            num_workers = num_workers,   
            shuffle = shuffle,         
            drop_last = False,
            collate_fn=collate_fn,)
    for (i_iter, input) in enumerate(loader):
        print(input['video'].size())
        print(input['label'].size(),input['label'].type())
        break