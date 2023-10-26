# ----------------
# Author: neonkill
# ----------------

import torch
import torch.nn as nn
import torchvision
from ..modules import Aggregator_LN_GELU, ConvBlock, LayerNorm, ConvNeXtBlock
# from model_module.model.backbones.seg_model import SegModel

class Backbone(nn.Module):
    def __init__(self, 
                seg_chs, 
                bev_chs,
                reduce_dim):
        super().__init__()

        depth_chs = seg_chs
        self.pretrained_convnext = torchvision.models.convnext_tiny(weights='IMAGENET1K_V1')
        self.pretrained_convnext.avgpool = nn.Identity()
        self.pretrained_convnext.classifier = nn.Identity()
        self.bev_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        self.depth_layers = nn.ModuleList()
        self.get_feats_layer_num = [1, 3, 5, 7]
        for ch in bev_chs:
            layer = nn.Sequential(nn.Conv2d(ch, reduce_dim, 1),
                                  LayerNorm(reduce_dim,  eps=1e-6, data_format = "channels_first"),
                                nn.GELU())
            self.bev_layers.append(layer)

        for ch in seg_chs:
            layer = nn.Sequential(nn.Conv2d(ch, reduce_dim, 1),
                                  LayerNorm(reduce_dim,  eps=1e-6, data_format = "channels_first"),
                                nn.GELU())
            self.seg_layers.append(layer)
            
        for ch in depth_chs:
            layer = nn.Sequential(nn.Conv2d(ch, reduce_dim, 1),
                                  LayerNorm(reduce_dim, eps=1e-6, data_format = "channels_first"),
                                nn.GELU())
            self.depth_layers.append(layer)
            
        self.layer64 = ConvNeXtBlock(768, 0.1)
        self.layergap = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            ConvBlock(768, 768, 1, 1, 0, norm='LN', act="GELU")
        )

        self.reduce_bev = Aggregator_LN_GELU(reduce_dim, bev_chs, target_idx=1)
        self.reduce_depth = Aggregator_LN_GELU(reduce_dim, depth_chs, target_idx=1)
        self.reduce_seg = Aggregator_LN_GELU(reduce_dim, bev_chs, target_idx=1)
        
        #! freeze
        # for param in self.pretrained_convnext.parameters():
        #     param.requires_grad_(False)

        # for param in self.gap.parameters():
        #     param.requires_grad_(False)
            
    def forward(self, x):
        '''
        Input: 6 RGB images (6b, 3, h, w)
        Output: Aggregated features (6b, C, h/32, w/32)
        '''
        feats = []
        for i in range(len(self.pretrained_convnext.features)):
            x = self.pretrained_convnext.features[i](x)
            if i in self.get_feats_layer_num:
                feats.append(x)
                
        f64 = self.layer64(x)
        feats.append(f64)
        gap = self.layergap(f64)
        feats.append(gap)

        bev_reds = []
        depth_reds = []
        seg_reds = []
        
        for i, feat in enumerate(feats):
            bev_reds.append(self.bev_layers[i](feat))
            depth_reds.append(self.depth_layers[i](feat))
            seg_reds.append(self.seg_layers[i](feat))
                        
            
        bev_agg = self.reduce_bev(bev_reds)
        depth_agg = self.reduce_depth(depth_reds)
        seg_agg = self.reduce_seg(seg_reds)
        
        return bev_agg, depth_agg, seg_agg