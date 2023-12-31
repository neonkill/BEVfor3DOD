from turtle import forward
from typing import OrderedDict
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class DefaultResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DefaultResBlock, self).__init__()
        self.convblock1 = ConvBlock(in_channels, out_channels, 3)
        self.conv3x3_1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.convblock1(x)
        out = self.conv3x3_1(out)
        out = self.bn_1(out)
        out += identity

        out = self.relu(out)

        return out

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

class Aggregator_LN_GELU(nn.Module):

    def __init__(self, in_dim, bins, target_idx):
        super(Aggregator_LN_GELU, self).__init__()

        self.target_idx = target_idx
        self.aggregator = ConvBlock(in_dim*len(bins), in_dim, 1, norm='LN', act='GELU')

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


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, norm = 'BN', act = "ReLU"):
        super(ConvBlock, self).__init__()
        if padding is None:
            pad = (kernel_size - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding=pad, bias=False)
        if norm == "BN":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "LN":
            self.norm = LayerNorm(out_channels, eps=1e-6, data_format = "channels_first")
        else:
            ValueError("norm init error")
        if act == "ReLU":
            self.act = nn.ReLU(inplace=True)
            
        elif act == "GELU":
            self.act = nn.GELU()
        else:
            ValueError("act init error")
            
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x