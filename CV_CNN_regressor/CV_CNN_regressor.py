#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 18:24:34 2025

@author: chenxinran
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, img_tensors, num_list, target_list):
        self.img_tensors = img_tensors
        self.num_list = num_list
        self.target_list = target_list

    def __len__(self):
        return len(self.img_tensors)

    def __getitem__(self, index):
        img = self.img_tensors[index]
        num = self.num_list[index]
        target = self.target_list[index]

        return img, num, target

class CNNRegressorCV(nn.Module):
    def __init__(self):
        super(CNNRegressorCV, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(5625, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x1 = self.pool(torch.relu(self.conv1(x)))
        x2 = self.pool(torch.relu(self.conv2(x1)))
        x3 = x2.view(x2.size(0), -1)
        x4 = torch.relu(self.fc1(x3))
        x5 = self.fc2(x4)
        return x5, x1, x2

def evaluate (model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []
    all_num = []
    with torch.no_grad():
       for image, num, label in dataloader:
           outputs, _, _ = model(image)
           all_labels.extend(label.numpy())
           all_predictions.extend(outputs.numpy())
           all_num.extend(num.numpy())
    return all_labels, all_predictions, all_num

def get_feature (model, dataloader):
    model.eval()
    fused_feats = []
    features2 = []
    feat_num = []
    with torch.no_grad():
       for image, num, label in dataloader:
           _, feature1, feature2 = model(image)
           fused_feat = torch.sum(torch.abs(feature1), dim=0)
           feature2 = feature2.squeeze()
           fused_feats.append(fused_feat)
           features2.append(feature2)
           feat_num.extend(num.numpy())
    return fused_feats, feat_num, features2
