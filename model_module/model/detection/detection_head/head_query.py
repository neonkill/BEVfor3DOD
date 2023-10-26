# ----------------
# Author: neonkill
# ----------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler)
from ..modules import ResBlock, LayerNorm, ConvNeXtBlock
from mmdet.models import build_loss
import copy

class HEAD(nn.Module):
    def __init__(self, num_class, in_channels, num_block, num_decoder_layers, pc_range):
        super().__init__()

        # self.transformer = transformer

        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.num_block = num_block
        self.cls_out_channels = num_class
        self.code_size = 10
        self.embed_dims = in_channels

        cls_branch = []
        for _ in range(self.num_block):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(LayerNorm(self.embed_dims, eps=1e-6, data_format = "channels_last"))
            cls_branch.append(nn.GELU())
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)
        self.cls_branches = nn.ModuleList([copy.deepcopy(cls_branch) for _ in range(num_decoder_layers)])

        reg_branch = []
        for _ in range(self.num_block):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.GELU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        self.reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(num_decoder_layers)])

    def forward(self, querys):
        # print(len(querys))
        # exit()
        batch_size, num_query, dim = querys[0].shape
        
        outputs_classes = []
        outputs_coords = []
        for i in range(len(querys)):
            
            outputs_class = self.cls_branches[i](querys[i])
            tmp = self.reg_branches[i](querys[i])

            # outputs_class = outputs_class.flatten(2)
            # tmp = tmp.flatten(2)
            
            # tmp[..., 0:2] += Xw_point
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])
            
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'all_cls_preds': outputs_classes,
            'all_bbox_preds': outputs_coords,
        }

        return outs



