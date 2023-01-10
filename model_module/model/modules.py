# ----------------
# Author: yelin2
# ----------------

from turtle import forward
from typing import OrderedDict
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, factor=2, norm='BN'):
        super().__init__()

        b_dim = out_ch // factor if (out_ch != 1) else out_ch
        
        if norm == 'BN':
            normalization = nn.BatchNorm2d
        elif norm == 'LN':
            normalization = nn.LayerNorm

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, b_dim, 3, padding=1, bias=False),
            normalization(b_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(b_dim, out_ch, 1, padding=0, bias=False),
            normalization(out_ch))

        self.up = nn.Conv2d(in_ch, out_ch, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        up = self.up(identity)
        x = x + up

        return self.relu(x)

        
# class Aggregator(nn.Module):

#     def __init__(self, in_dim, bins):
#         super(Aggregator, self).__init__()

#         self.aggregator = nn.Sequential(
#             nn.Conv2d(in_dim*len(bins), in_dim, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_dim),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, target, feats):
#         _,_,H,W = target.size() # B,C,H,W
        
#         sum_feats=[target]
#         for feat in feats:
#             sum_feats.append(F.interpolate(feat, (H,W), mode='bilinear', align_corners=True))

#         context = self.aggregator(torch.cat(sum_feats, dim=1))

#         return context


class Aggregator(nn.Module):

    def __init__(self, in_dim, bins, target_idx):
        super(Aggregator, self).__init__()

        self.target_idx = target_idx
        self.aggregator = nn.Sequential(
            nn.Conv2d(in_dim*len(bins), in_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats):

        _,_,H,W = feats[self.target_idx].size() # B,C,H,W
        
        sum_feats=[]
        for i, feat in enumerate(feats):
            if i < self.target_idx:
                sum_feats.append(F.adaptive_avg_pool2d(feat, (H,W)))                
            elif i == self.target_idx:
                sum_feats.append(feat)
            elif i > self.target_idx:
                sum_feats.append(F.interpolate(feat, (H,W), mode='bilinear', align_corners=True))

        context = self.aggregator(torch.cat(sum_feats, dim=1))

        return context