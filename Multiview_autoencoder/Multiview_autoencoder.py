#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:20:14 2025

@author: chenxinran
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MultiviewImageDataset(Dataset):
    def __init__(self, img_tensors, img_tensors_sub, img_tensors_cat, num_list):
        self.img_tensors = img_tensors
        self.img_tensors_sub = img_tensors_sub
        self.img_tensors_cat = img_tensors_cat
        self.num_list = num_list

    def __len__(self):
        return len(self.img_tensors)

    def __getitem__(self, index):
        img = self.img_tensors[index]
        img_sub = self.img_tensors_sub[index]
        img_cat = self.img_tensors_cat[index]
        num = self.num_list[index]

        return img, img_sub, img_cat, num

class MultiViewConvAutoencoder(nn.Module):
    def __init__(self):
        super(MultiViewConvAutoencoder, self).__init__()
        self.encoder1 = self._build_encoder()
        self.encoder2 = self._build_encoder()
        self.encoder3 = self._build_encoder()
        
        self.shared_fc = nn.Linear(1 * 3, 1)
        
        self.decoder1 = self._build_decoder()
        self.decoder2 = self._build_decoder()
        self.decoder3 = self._build_decoder()
        
    def _build_encoder(self):

        return nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(625, 1)
        )
    
    def _build_decoder(self):

        return nn.Sequential(
            nn.Linear(1, 625),
            nn.ReLU(True),
            nn.Unflatten(1, (25, 5, 5)),
            nn.ConvTranspose2d(25, 128, 4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.Upsample(size=(400, 400), mode='bilinear', align_corners=False), 
            nn.Sigmoid()  
       )

    
    def forward(self, img1,img2, img3):
        code1 = self.encoder1(img1)
        code2 = self.encoder2(img2)
        code3 = self.encoder3(img3)
        
        shared_code = self.shared_fc(torch.cat([code1, code2, code3], dim=1))
        
        rec1 = self.decoder1(shared_code)
        rec2 = self.decoder2(shared_code)
        rec3 = self.decoder3(shared_code)
        rec1 = rec1.mean(dim=0, keepdim=True)
        rec2 = rec2.mean(dim=0, keepdim=True)
        rec3 = rec3.mean(dim=0, keepdim=True)
        
        return rec1, rec2, rec3, shared_code

def extract_features(model, data_loader, device):
    shared_features = []
    num_list =[]
    model.eval()
    with torch.no_grad():
        for img1, img2, img3, num in data_loader:
            img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)
            num_list.append(num)
            _, _, _, shared_feat = model(img1, img2, img3)
            shared_features.append(shared_feat)
    shared_features = torch.cat(shared_features, dim=1) 
    shared_features = shared_features.reshape(len(num_list), -1)
            
    return shared_features, num_list