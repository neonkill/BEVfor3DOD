# ----------------
# Author: yelin2
# ----------------

import torch
import torch.nn as nn
from einops import rearrange

from model_module.model.modules import Aggregator



class FullModel(torch.nn.Module):

    def __init__(self, backbone, seg_extractor, seg_head, 
                det_extractor=None, det_head=None):
        super().__init__()

        self.backbone = backbone
        self.seg_extractor = seg_extractor
        self.seg_head = seg_head

        

    def forward(self, x):
        b, n, _, _, _ = x['image'].shape
        imgs = rearrange(x['image'], 'b n ... -> (b n) ...')           # (b n) c h w

        # agg4, agg16 = self.backbone(imgs)                              # (b n) c h w
        agg16 = self.backbone(imgs)                              # (b n) c h w

        I = x['intrinsics']                # b 3 3
        E = x['extrinsics']                # b 4 4
        agg16 = rearrange(agg16, '(b n) ... -> b n ...', b=b, n=n)     # b n c h w
        seg_feat = self.seg_extractor(agg16, I, E)
        seg_maps = self.seg_head(seg_feat)

        out = {k: pred[:, start:stop] for k, (start, stop) in self.outputs.items()}
        
        return out
        
        # return {'bev': seg_maps}
