'''
Author: yelin2
'''

import torch
import torch.nn as nn
from ..modules import Aggregator


class Backbone(nn.Module):
    def __init__(self, 
                efficientnet, 
                chs, 
                reduce_dim,
                extra_layers=None):
        super().__init__()

        # Feature extractor
        self.efficientnet = efficientnet
        self.chs = chs

        self.layers = nn.ModuleList()
        for ch in chs:
            layer = nn.Conv2d(ch, reduce_dim, 1)
            self.layers.append(layer)
        
        ex_idx = len(chs) - len(extra_layers)
        self.B64 = nn.Sequential(nn.Sequential(
                                nn.Conv2d(chs[ex_idx], chs[ex_idx], 
                                        kernel_size=3, stride=2,
                                        bias=False),
                                nn.BatchNorm2d(chs[ex_idx]),
                                nn.ReLU(inplace=True)))

        self.GAP = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(chs[ex_idx+1], chs[ex_idx+1], 
                                        kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(chs[ex_idx+1]),
                                nn.ReLU(inplace=True))

        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        # target_idx = chs에서 target resolution의 index
        self.reduce = Aggregator(reduce_dim, chs, target_idx=0)
        
            
    def forward(self, x):
        '''
        Input: 6 RGB images (6b, 3, h, w)
        Output: Aggregated features (6b, C, h/32, w/32)
        '''

        reds = []
        feats = self.efficientnet(x)  # 4, 8, 16, 32
        F64 = self.B64(feats[-1])
        GAP = self.GAP(F64)
        feats.extend([F64, GAP])        # 4, 8, 16, 32, 64, GAP

        
        for i, feat in enumerate(feats):
            #! debugging
            # print(f'{2**(i+2)} res : {feat.shape}')
            reds.append(self.layers[i](feat))

        agg = self.reduce(reds)
        # print(f'aggregated feats: {agg.shape}')
        return agg
        