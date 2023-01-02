# ----------------
# Author: yelin2
# ----------------

from turtle import forward
from typing import OrderedDict
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):

    def __init__(self, in_dim, bins):
        super(Aggregator, self).__init__()

        self.aggregator = nn.Sequential(
            nn.Conv2d(in_dim*len(bins), in_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, target, feats):
        _,_,H,W = target.size() # B,C,H,W
        
        sum_feats=[target]
        for feat in feats:
            sum_feats.append(F.interpolate(feat, (H,W), mode='bilinear', align_corners=True))

        context = self.aggregator(torch.cat(sum_feats, dim=1))

        return context