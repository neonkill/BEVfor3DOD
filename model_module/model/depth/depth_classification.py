import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
try:
    from model_module.model.modules import *
except:
    import sys
    sys.path.append('../')
    from modules import *
    

class DepthClassification(nn.Module):
    def __init__(self):
        super(DepthClassification, self).__init__()
        self.agg_size = 64
        self.depth_channels = 112 #bin
        
        self.each_1x1convs = nn.ModuleList()
        self.each_1x1convs.append(ConvBlock(64, self.agg_size, 1)) #1/4
        self.each_1x1convs.append(ConvBlock(128, self.agg_size, 1)) #1/8
        self.each_1x1convs.append(ConvBlock(256, self.agg_size, 1)) #1/16
        self.each_1x1convs.append(ConvBlock(512, self.agg_size, 1)) #1/32
        
        self.agg_conv = nn.Sequential(
                                ConvBlock(self.agg_size * 4, self.agg_size, kernel_size=1, stride=1, padding=0)
                            )
        
        self.pred_depth = nn.Sequential(
                                DefaultResBlock(self.agg_size, self.agg_size),
                                DefaultResBlock(self.agg_size, self.agg_size),
                                nn.Conv2d(self.agg_size, self.depth_channels, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, agg_feats):
        feats_4 = self.each_1x1convs[0](agg_feats[0])
        agg_size = feats_4.shape[2:]

        feats_8 = self.each_1x1convs[1](agg_feats[1])
        feats_8 = F.interpolate(feats_8, agg_size, mode = 'bilinear')

        feats_16 = self.each_1x1convs[2](agg_feats[2])
        feats_16 = F.interpolate(feats_16, agg_size, mode = 'bilinear')

        feats_32 = self.each_1x1convs[3](agg_feats[3])
        feats_32 = F.interpolate(feats_32, agg_size, mode = 'bilinear')
        
        agg_feats = torch.cat([feats_4, feats_8, feats_16, feats_32], dim=1)
        
        agg_feats = self.agg_conv(agg_feats)

        depth_bin = self.pred_depth(agg_feats)
        
        return depth_bin