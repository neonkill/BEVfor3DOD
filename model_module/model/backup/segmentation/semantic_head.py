# ----------------
# Author: yelin2
# ----------------

import torch
import torch.nn as nn
from model_module.model.modules import ResBlock

'''

'''

class SemanticHead(torch.nn.Module):

    def __init__(self, output, dim=64):
        super().__init__()

        self.up = nn.Sequential(
                            nn.Upsample(scale_factor=2, 
                                        mode='bilinear', 
                                        align_corners=True),
                            nn.Conv2d(dim, dim, 
                                    3, padding=1, bias=False),
                            nn.BatchNorm2d(dim),
                            nn.ReLU(inplace=True))
        
        self.resblock = nn.Sequential(
                                ResBlock(in_ch=dim, out_ch=dim),     
                                ResBlock(in_ch=dim, out_ch=int(dim/2)))

        self.to_logits = nn.Sequential(
                                nn.Conv2d(int(dim/2), int(dim/2), 3, padding=1, bias=False),
                                nn.BatchNorm2d(int(dim/2)),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(int(dim/2), len(output.keys()), 1))

    def forward(self, x):
        '''
        Input: [B 64 128 128]
        Ourput: [B 2 256 256]
        '''

        x = self.up(x)
        x = self.resblock(x)
        x = self.to_logits(x)

        return x
