# ----------------
# Author: neonkill
# ----------------

import torch.nn as nn
from einops import rearrange
from .modules import ResBlock, LayerNorm, ConvNeXtBlock


class FullModel(nn.Module):
    def __init__(self, backbone, 
                q_generator,
                bev_generator,
                det_head,
                reduce_dim, 
                norm='BN',
                outputs={'bev': [0, 1]}):
        '''
        image_height, image_width: input RGB size
        b_res: extracted feature resolution from backbone (32 or 128)
        '''

        super(FullModel, self).__init__()


        self.backbone = backbone
        self.q_generator = q_generator
        self.bev_generator = bev_generator
        self.det_head = det_head

        self.num_class = 13
        self.depth_channels = 112
        dim = reduce_dim
        if norm == 'BN':
            self.depth_pred = nn.Sequential(ResBlock(in_ch=dim, out_ch=dim, norm=norm),     
                                    ResBlock(in_ch=dim, out_ch=dim, norm=norm),
                                    nn.Conv2d(dim, self.depth_channels, kernel_size=1, stride=1, padding=0))
            
            self.pred = nn.Sequential(ResBlock(in_ch=dim, out_ch=dim, norm=norm),     
                                    ResBlock(in_ch=dim, out_ch=int(dim/2), norm=norm))

            self.to_logits = nn.Sequential(
                                nn.Conv2d(int(dim/2), int(dim/2), 3, padding=1, bias=False),
                                nn.BatchNorm2d(int(dim/2)),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(int(dim/2), len(outputs.keys()), 1))
            
        elif norm == 'LN':
            self.depth_pred = nn.Sequential(ConvNeXtBlock(dim, 0.0),     
                                    ConvNeXtBlock(dim, 0.0),
                                    nn.Conv2d(dim, self.depth_channels, kernel_size=1, stride=1, padding=0))
            
            self.seg_pred = nn.Sequential(ConvNeXtBlock(dim, 0.0),     
                                    ConvNeXtBlock(dim, 0.0),
                                    nn.Conv2d(dim, self.num_class, kernel_size=1, stride=1, padding=0))
            
            # self.pred_seg = nn.Sequential(ConvNeXtBlock(48, 0.0),     
            #                         ConvNeXtBlock(48, 0.0))
            
            self.pred_det = nn.Sequential(ConvNeXtBlock(256, 0.0),     
                                    ConvNeXtBlock(256, 0.0))
            
            # self.to_logits = nn.Sequential(
            #                     nn.Conv2d(48, int(48/2), 3, padding=1, bias=False),
            #                     LayerNorm(int(48/2), eps=1e-6, data_format = "channels_first"),
            #                     nn.GELU(),
            #                     nn.Conv2d(int(48/2), len(outputs.keys()), 1))

        self.outputs = outputs


    def forward(self, sample):
        '''
        Sample(dict)
            - cam_idx ([16, 6])
            - image ([1, 6, 3, 900, 1600])
            - intrinsics ([16, 6, 3, 3])
            - extrinsics ([16, 6, 4, 4])
            - bev ([16, 12, 200, 200])
            - view ([16, 3, 3])
            - visibility ([16, 200, 200])
            - pose ([16, 4, 4])        
        '''

        b, n, _, _, _ = sample['image'].shape


        # Extract 1/8, 1/128(or 1/32) feature from RGB image        
        imgs = rearrange(sample['image'], 'b n ... -> (b n) ...')  # (b n) c h w
        F128_flat, depth_feats, seg_feats = self.backbone(imgs)                            # (b n) c h w
        
        depth_bin = self.depth_pred(depth_feats)
        depth_bin = depth_bin.softmax(1)
        
        seg_logits = self.seg_pred(seg_feats)
        
        F128 = rearrange(F128_flat, 
                            '(b n) ... -> b n ...', b=b, n=n)      # b n c h w


        # Generate Query
        I = sample['intrinsics']                # b 3 3
        E = sample['extrinsics']                # b 4 4
        bev_8 = self.q_generator(F128, I, E)      # b d bh bw
        bev = self.bev_generator(F128, I, E, bev_8)
        # bev_seg = self.pred_seg(bev_8)                    # b 2 bh bw
        # pred = self.to_logits(bev_seg)
        
        # bev_8 = bev_8.detach()

        bev_det = self.pred_det(bev) 
        det_pred = self.det_head(bev_det)

        out = dict()
        # out = {k: pred[:, start:stop] for k, (start, stop) in self.outputs.items()}
        
        out['depth_bin'] = depth_bin
        out['seg_logits'] = seg_logits
        out['det_pred'] = det_pred
        return out
