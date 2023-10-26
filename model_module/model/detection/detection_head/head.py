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
    def __init__(self, num_class, in_channels, num_block, num_decoder_layers, pc_range, bev_embedding):
        super().__init__()

        # self.transformer = transformer

        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.num_block = num_block
        self.cls_out_channels = num_class
        self.code_size = 10
        self.embed_dims = in_channels
        self.bev_h = 25
        self.bev_w = 25
        self.bev_embedding = BEVEmbedding(**bev_embedding)
        self.bev_embed = nn.Conv2d(2, self.embed_dims, 1)

        cls_branch = []
        for _ in range(self.num_block):
            cls_branch.append(nn.Conv2d(self.embed_dims, self.embed_dims,1))
            cls_branch.append(LayerNorm(self.embed_dims, eps=1e-6, data_format = "channels_first"))
            cls_branch.append(nn.GELU())
        cls_branch.append(nn.Conv2d(self.embed_dims, self.cls_out_channels,1))
        self.cls_branch = nn.Sequential(*cls_branch)
        

        reg_branch = []
        for _ in range(self.num_block):
            reg_branch.append(nn.Conv2d(self.embed_dims, self.embed_dims, 1))
            reg_branch.append(nn.GELU())
        reg_branch.append(nn.Conv2d(self.embed_dims, self.code_size, 1))
        self.reg_branch = nn.Sequential(*reg_branch)
        

    def forward(self, bev_feat):
        batch_size, channels, bh, bw = bev_feat.shape
        
        Xw = self.bev_embedding.grid[:2]
        bev_embed = self.bev_embed(Xw[None])

        bev_feat_embed = bev_feat + bev_embed
        # output = bev_feat_embed
        Xw = Xw.flatten(1).unsqueeze(0).repeat(batch_size, 1,1).permute(0,2,1)
        
        outputs_classes = []
        outputs_coords = []

        outputs_class = self.cls_branch(bev_feat_embed)
        tmp = self.reg_branch(bev_feat_embed)
        
        outputs_class = outputs_class.flatten(2).permute(0,2,1)
        tmp = tmp.flatten(2).permute(0,2,1)
        
        tmp[..., 0:2] += Xw
        # tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        # tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
        # tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
        #                  self.pc_range[0]) + self.pc_range[0])
        # tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
        #                  self.pc_range[1]) + self.pc_range[1])
        # tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
        #                  self.pc_range[2]) + self.pc_range[2])
        
        outputs_coord = tmp
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
        # outputs_classes.append(outputs_class.permute(0,2,1))
        # outputs_coords.append(outputs_coord.permute(0,2,1))

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'all_cls_preds': outputs_classes,
            'all_bbox_preds': outputs_coords,
        }

        return outs



class BEVEmbedding(nn.Module):
    def __init__(
        self,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        resolution: int,
    ):
        """
        Only real arguments are:

        resolution: bev resolution (ex 1/4: 4)

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        self.bev_height = bev_height
        self.bev_width = bev_width
        
        # each decoder block upsamples the bev embedding by a factor of 2
        assert (resolution % 2 == 0) or 1, 'resolution arg must devided into 2'
        h = bev_height // resolution
        w = bev_width // resolution

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    #! pytorch 1.8.0에는 torch.meshgrid의 indexing param 지원 안함...
    #! indexing이 'xy'가 아닌 'ij'로 설정되어서 meshgrid 만들때 x, y의 좌표를 바꿔서 넣어줘야 함
    #! meshgrid의 좌표가 x y가 아닌 y x로 출력되기 때문에 stack 순서도 바꿔줘야 함!!
    #! indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    y, x = torch.meshgrid((ys, xs))
    indices = torch.stack([x, y], 0)
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


