# encoding: utf-8
import cv2
import torch

import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    return video        


target_dir = '/home/st392/code/datasets/LRW/lrw_LipCropInside_pkl'

if(not os.path.exists(target_dir)):
    os.makedirs(target_dir)    

class LRWDataset(Dataset):
    def __init__(self):

        with open('/home/st392/code/learn-an-effective-lip-reading-model-without-pains/label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()            
        
        self.list = []

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join('/home/st392/code/datasets/LRW/lipCropInside_mp4', label, '*', '*.mp4'))
            for file in files:
                savefile = file.replace('/home/st392/code/datasets/LRW/lipCropInside_mp4', target_dir).replace('.mp4', '.pkl')
                savepath = os.path.split(savefile)[0]
                if(not os.path.exists(savepath)):
                    os.makedirs(savepath)
                
            files = sorted(files)
            

            self.list += [(file, i) for file in files]                                                                                
            
        
    def __getitem__(self, idx):
        savename = self.list[idx][0].replace('/home/st392/code/datasets/LRW/lipCropInside_mp4', target_dir).replace('.mp4', '.pkl')    
        inputs = extract_opencv(self.list[idx][0])
        result = {}        
         
        name = self.list[idx][0]
        duration = self.list[idx][0]            
        labels = self.list[idx][1]
        result['video'] = inputs
        result['label'] = int(labels)
        result['duration'] = self.load_duration(duration.replace('.mp4', '.txt').replace("lipCropInside_mp4","lipread_mp4")).astype(bool)
        torch.save(result, savename)
        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])
        
        tensor = np.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor            

if(__name__ == '__main__'):
    loader = DataLoader(LRWDataset(),
            batch_size = 96, 
            num_workers = 16,   
            shuffle = False,         
            drop_last = False)
    
    import time
    tic = time.time()
    for batch in tqdm(loader):
        pass