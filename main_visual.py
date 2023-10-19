import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
import numpy as np
import time
from model import *
import torch.optim as optim 
import random
import pdb
import shutil
from LSR import LSR
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler


torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser.add_argument('--gpus', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--n_class', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_epoch', type=int, required=True)
parser.add_argument('--test', type=str2bool, required=True)

# load opts
parser.add_argument('--weights', type=str, required=False, default=None)

# save prefix
parser.add_argument('--save_prefix', type=str, required=True)

# dataset
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--datasetVersion', type=str, required=True)
parser.add_argument('--border', type=str2bool, required=True)
parser.add_argument('--mixup', type=str2bool, required=True)
parser.add_argument('--label_smooth', type=str2bool, required=True)
parser.add_argument('--se', type=str2bool, required=True)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

if(args.dataset == 'lrw'):
    from utils import LRWDataset as Dataset
    from utils import LRW_collate as collate_fn
elif(args.dataset == 'lrw1000'):    
    from utils import LRW1000_Dataset as Dataset
else:
    raise Exception('lrw or lrw1000')    

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

video_model = VideoModel(args).to(device)

def parallel_model(model):
    model = nn.DataParallel(model)
    return model        


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
    

lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
optim_video = optim.Adam(video_model.parameters(), lr = lr, weight_decay=1e-4)     
scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_video, T_max = args.max_epoch, eta_min=5e-6)


if(args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))    
    load_missing(video_model, weight.get('video_model'))
    
          
video_model = parallel_model(video_model)

def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    print(len(dataset), batch_size, num_workers, shuffle)
    loader =  DataLoader(dataset,
            batch_size = batch_size, 
            num_workers = num_workers,   
            shuffle = shuffle,         
            drop_last = False,
            pin_memory=True,
            collate_fn=collate_fn)
    return loader
valdataset = Dataset('val', args)
valloader = dataset2dataloader(valdataset, args.batch_size, args.num_workers, shuffle=False)  
def add_msg(msg, k, v):
    if(msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg    

def test():
    from tqdm import tqdm
    with torch.no_grad():
        print('Start Testing, Data Length:',len(valdataset))
        
        print('start testing')
        v_acc = []
        v_loss = []
        entropy = []
        acc_mean = []
        total = 0
        cons_acc = 0.0
        cons_total = 0.0
        attns = []
        lossF = nn.CrossEntropyLoss()

        for (i_iter, input) in tqdm(enumerate(valloader)):
            
            video_model.eval()
            
            tic = time.time()
            video,label = input
            video = input.get('video').to(device)
            label = input.get('label').to(device)
            total = total + video.size(0)
            # names = input.get('name')
            # border = input.get('duration').to(device).float()
            
            with autocast():
                # if(args.border):
                #     y_v = video_model(video, border)                                           
                # else:
                y_v = video_model(video)                                           
                                
            v_acc.extend((y_v.argmax(-1) == label.argmax(-1)).cpu().numpy().tolist())
            print(lossF(y_v, label).cpu().numpy().tolist())
            v_loss.extend(lossF(y_v, label).cpu().numpy().tolist())
            toc = time.time()
            if(i_iter % 10 == 0):  
                msg = ''              
                msg = add_msg(msg, 'v_acc={:.5f}', np.array(v_acc).reshape(-1).mean())                
                msg = add_msg(msg, 'eta={:.5f}', (toc-tic)*(len(valloader)-i_iter)/3600.0)
                                
                print(msg)         
               

        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = 'v_acc_{:.5f}_'.format(acc)
        loss = float(np.array(v_loss).reshape(-1).mean())
        return acc, msg,loss

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)

def train():            
    
    import datetime
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    savePath = Path(f"/home/st392/compute/LRW/{args.datasetVersion}/{args.lr}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    savePath.mkdir(parents=True, exist_ok=True)
    
    dataset = Dataset('train', args)
    print('Start Training, Data Length:',len(dataset))
    
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
        
    max_epoch = args.max_epoch    

    
    ce = nn.CrossEntropyLoss()

    tot_iter = 0
    best_acc = 0.0
    best_epoch = 0
    adjust_lr_count = 0
    trainLoss = []
    trainAcc = []
    testLoss = []
    testAcc = []
    alpha = 0.2
    beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
    scaler = GradScaler()     
    loop = tqdm(max_epoch*len(loader))    
    loop.set_description('Start Training')    
    for epoch in range(max_epoch):
        total = 0.0
        v_acc = 0.0
        total = 0.0               
        
        lsr = LSR()
        
        
        for (i_iter, input) in enumerate(loader):
            tic = time.time()           
            
            video_model.train()
            video = input.get('video').to(device)
            label = input.get('label').to(device)
            
            loss = {}
            
            if(args.label_smooth):
                loss_fn = lsr
            else:
                loss_fn = nn.CrossEntropyLoss()
            
            with autocast():
                if(args.mixup):
                    lambda_ = np.random.beta(alpha, alpha)
                    index = torch.randperm(video.size(0)).to(device)
                    
                    mix_video = lambda_ * video + (1 - lambda_) * video[index, :]
                        
                    label_a, label_b = label, label[index]            

                    if(args.border):
                        # y_v = video_model(mix_video, mix_border)   
                        pass    
                    else:                
                        y_v = video_model(mix_video)       

                    loss_bp = lambda_ * loss_fn(y_v, label_a) + (1 - lambda_) * loss_fn(y_v, label_b)
                    
                else:
                    if(args.border):
                        # y_v = video_model(video, border)
                        pass       
                    else:                
                        y_v = video_model(video)    
                    loss_bp = loss_fn(y_v, label)
                                    
                #plot loss and accuracy
                plt.clf()
                plt.subplot(121)
                plt.plot(np.arange(len(trainAcc)), trainAcc, label='Train Accuracy')
                plt.plot(np.arange(len(trainAcc)), testAcc, label='Val Accuracy')
                plt.legend()
                plt.xlabel("Batch")
                plt.ylabel("Accuracy")
                plt.title("Accuracy Plots")
                plt.subplot(122)
                plt.plot(np.arange(len(trainLoss)), trainLoss, label='Train Loss')
                plt.plot(np.arange(len(trainLoss)), testLoss, label='Val Loss')
                plt.legend()
                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.title("Loss Plots")
                plt.savefig(savePath / f'loss.jpg')

            loss['CE V'] = loss_bp
                
            optim_video.zero_grad()   
            scaler.scale(loss_bp).backward()  
            scaler.step(optim_video)
            scaler.update()
            
            toc = time.time()
            trainLoss.append(loss_bp.item())
            trainAcc.append((y_v.argmax(-1) == label.argmax(-1)).cpu().numpy().tolist())

            

            if(i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0)):

                acc, msg,loss = test()
                testLoss.append(loss)
                testAcc.append(acc)

                if(acc > best_acc):
                    savename = '{}/iter_{}_epoch_{}_{}.pt'.format(savePath,args.save_prefix, tot_iter, epoch, msg)
                    
                    temp = os.path.split(savename)[0]
                    if(not os.path.exists(temp)):
                        os.makedirs(temp)                    
                    torch.save(
                        {
                            'video_model': video_model.module.state_dict(),
                        }, savename)         


                if(tot_iter != 0):
                    best_acc = max(acc, best_acc)   
                    best_epoch = epoch if(acc > best_acc) else best_epoch 
            else:
                testLoss.append(testLoss[-1])
                testAcc.append(testAcc[-1])

            msg = 'epoch={},train_iter={}'.format(epoch, tot_iter,)
            for k, v in loss.items():                                                
                msg += ',{}={:.5f}'.format(k, v)
            msg = msg + str(',lr=' + str(showLR(optim_video)))                    
            msg = msg + str(',best_acc={:2f}'.format(best_acc))
            msg = msg + str(',best_epoch={:2f}'.format(best_epoch))
            loop.set_description(msg)
            loop.update(1)
            tot_iter += 1        
            
        scheduler.step()            
        
if(__name__ == '__main__'):
    if(args.test):
        acc, msg = test()
        print(f'acc={acc}')
        exit()
    train()
