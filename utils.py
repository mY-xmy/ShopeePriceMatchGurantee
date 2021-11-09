#!/usr/bin/env python
# coding=utf-8
"""
@FilePath: utils.py
@Author: Xu Mingyu
@Date: 2021-11-06 14:28:33
@LastEditTime: 2021-11-09 15:33:42
@Description: 
@Copyright 2021 Xu Mingyu, All Rights Reserved. 
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import _LRScheduler

from PIL import Image
import warnings

class ShopeeTrainDataset(Dataset):
    """ Self-defined dataset object for images """
    
    def __init__(self, df, transform=None):
        super(ShopeeTrainDataset, self).__init__()
        self.df = df
        class_mapping = {group : i for i, group in enumerate(set(self.df['label_group']))}
        self.df['label_class'] = self.df['label_group'].map(class_mapping)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image_path = self.df.image_path.iloc[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.df.label_class.iloc[index]
        return image, label


class ShopeeImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, index):
        image_path = self.dataset.image_path.iloc[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image

class ShopeeScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_start=5e-6, lr_max=1e-5,
                 lr_min=1e-6, lr_warmup_ep=5, lr_sus_ep=0, lr_decay=0.4, step_size=1,
                 last_epoch=-1):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_warmup_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        self.step_size = step_size
        super(ShopeeScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            self.last_epoch += 1
            return [self.lr_start for _ in self.optimizer.param_groups]
        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1
        return [lr for _ in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        return self.base_lrs
    
    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = ((self.lr_max - self.lr_start) / 
                  self.lr_ramp_ep * self.last_epoch + 
                  self.lr_start)
        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max
        else:
            lr = ((self.lr_max - self.lr_min) * self.lr_decay**
                  ((self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep + self.step_size - 1) // self.step_size) + 
                  self.lr_min)
        return lr

def euclidean_dist(x, y, norm=False):
    m, n = x.size(0), y.size(0)
    
    if norm:
        x = x / x.norm(p=2, dim=1, keepdim=True)
        y = y / y.norm(p=2, dim=1, keepdim=True)
    
    xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def cosine_dist(x,y):
    m, n = x.size(0), y.size(0)
    
    norm_x = x.norm(p=2, dim=1, keepdim=True).expand(m,n)
    norm_y = y.norm(p=2, dim=1, keepdim=True).expand(n,m).t()
    dist = torch.matmul(x, y.t()) / (norm_x * norm_y)
    return dist

def DistancePredict(df, features, threshold = 0.9, chunk = 1024, max_preds=50, distance_type="cosine"):
    assert(distance_type in ("cosine","euclidean"))
    
    predict = []
    n = (features.size(0) + chunk - 1) // chunk
    with torch.no_grad():
        for i in range(n):
            a = i*chunk
            b = (i+1)*chunk
            b = min(b, features.size(0))
            x = features[a:b]
            y = features

            if distance_type =="cosine":
                distance = cosine_dist(x,y)
            elif distance_type == "euclidean":
                distance = euclidean_dist(x,y, norm=True)

            for k in range(b-a):
                if distance_type == "euclidean":
                    dist_asc = torch.sort(distance[k], descending=False)
                    idx = dist_asc[1][dist_asc[0] < threshold][:max_preds].detach().cpu().numpy()
                else:
                    dist_desc = torch.sort(distance[k], descending=True)
                    idx = dist_desc[1][dist_desc[0] > threshold][:max_preds].detach().cpu().numpy()
                    
                pred = df.iloc[idx].posting_id.to_numpy()
                predict.append(pred)
            del x,y,distance
            
    return predict

def f1(target, predict):
    n = len(np.intersect1d(target,predict))
    return 2*n/(len(target)+len(predict))

def precision(target, predict):
    n = len(np.intersect1d(target,predict))
    return n / len(predict)

def recall(target, predict):
    n = len(np.intersect1d(target,predict))
    return n / len(target)

def get_metric(target, predict):
    tmp = pd.DataFrame({"target":target.reset_index(drop=True), "predict":predict.reset_index(drop=True)})
    f1_score = tmp.apply(lambda row: f1(row['target'], row["predict"]),axis=1)
    precision_score = tmp.apply(lambda row: precision(row['target'], row["predict"]),axis=1)
    recall_score = tmp.apply(lambda row: recall(row['target'], row["predict"]),axis=1)
    #print("Mean F1: {:f}".format(f1_score.mean()))
    #print("Mean Precision: {:f}".format(precision_score.mean()))
    #print("Mean Recall: {:f}".format(recall_score.mean()))
    return f1_score.mean(), precision_score.mean(), recall_score.mean()

def validate(feature, threshold, df):
    pred = DistancePredict(df, feature, threshold= threshold)
    df["pred"] = pred
    f1, prec, rec = get_metric(df["target"], df["pred"])
    return f1, prec, rec

def NDCG(features, df, chunk=1024):
    df = df.reset_index()
    index_group_dict = df.groupby("label_group").index.agg("unique").to_dict()

    ndcg = np.zeros(features.shape[0])
    n = (features.size(0) + chunk - 1) // chunk
    with torch.no_grad():
        for i in range(n):
            a = i*chunk
            b = (i+1)*chunk
            b = min(b, features.size(0))
            x = features[a:b]
            y = features

            distance = cosine_dist(x,y).detach().cpu()
            for k in range(b-a):
                dist_desc = torch.sort(distance[k], descending=True)
                dist_desc_idx = dist_desc[1].numpy()

                target_index = index_group_dict[df.iloc[a+k].label_group]
                target_pos_index = np.argwhere(np.in1d(dist_desc_idx, target_index)).flatten()

                dcg = np.sum(1 / np.log2(target_pos_index + 2))
                idcg = np.sum(1 / np.log2(np.arange(1, target_index.shape[0]+1) + 1))
                ndcg[a+k] = dcg / idcg
    
    return ndcg