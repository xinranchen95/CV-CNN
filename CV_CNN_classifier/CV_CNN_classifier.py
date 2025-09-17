#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:30:10 2025

@author: chenxinran
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class FeatsDataset(Dataset):
    def __init__(self, features, num_list, target_list):
        self.features = features
        self.num_list = num_list
        self.target_list = target_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feat = self.features[index]
        num = self.num_list[index]
        target = self.target_list[index]

        return feat, num, target
    
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(8, 128)#自编码encodeing的维度
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate (model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_num = []
    with torch.no_grad():
       for feat, num, label in dataloader:
           feat = feat.to(device, dtype=torch.float32)  # 确保输入是 float32 并在 MPS 上
           label = label.to(device)
           outputs = model(feat)
           _, predicted = torch.max(outputs.data, 1)
           all_labels.extend(label.cpu().numpy())#mps设备不支持tensor转numpy，转到cpu上
           all_predictions.extend(predicted.cpu().numpy())
           all_num.extend(num.numpy())
    return all_labels, all_predictions, all_num