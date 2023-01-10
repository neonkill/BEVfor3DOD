# ----------------
# Author: yelin2
# ----------------

import torch.nn as nn
from einops import rearrange
from .modules import ResBlock


class FullModel(nn.Module):
    def __init__(self, backbone, 
                q_generator, 
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

        dim = reduce_dim
        normalization = nn.BatchNorm2d

        self.pred = nn.Sequential(ResBlock(in_ch=dim, out_ch=dim, norm=norm),     
                                ResBlock(in_ch=dim, out_ch=int(dim/2), norm=norm))
        self.to_logits = nn.Sequential(
                                nn.Conv2d(int(dim/2), int(dim/2), 3, padding=1, bias=False),
                                normalization(int(dim/2)),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(int(dim/2), len(outputs.keys()), 1))

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
        F128_flat = self.backbone(imgs)                            # (b n) c h w
        F128 = rearrange(F128_flat, 
                            '(b n) ... -> b n ...', b=b, n=n)      # b n c h w


        # Generate Query
        I = sample['intrinsics']                # b 3 3
        E = sample['extrinsics']                # b 4 4
        bev = self.q_generator(F128, I, E)      # b d bh bw

        # print(bev.shape)
        bev = self.pred(bev)                    # b 2 bh bw
        pred = self.to_logits(bev)
        # print(pred.shape)
        out = {k: pred[:, start:stop] for k, (start, stop) in self.outputs.items()}

        return out
