# ----------------
# Author: yelin2
# ----------------

import math
import torch
import torch.nn as nn
from einops import rearrange


'''

'''

class SemanticFeatureExtractor(torch.nn.Module):

    def __init__(self, embed_dims=64, n_head=4, q_size=16, full_q_size=128):
        super().__init__()

        self.q_size = q_size
        self.q_generator = QueryGenerator(q_size=q_size, embed_dims=embed_dims)

        self.match = nn.TransformerDecoderLayer(d_model=embed_dims, 
                                                nhead=n_head, 
                                                dim_feedforward=embed_dims*2, 
                                                batch_first=True)

        


        layers = []
        bev_resolution = int(full_q_size/q_size)

        for _ in range(int(math.log2(bev_resolution))):
            layers.append(nn.Sequential(
                            nn.Upsample(scale_factor=2, 
                                        mode='bilinear', 
                                        align_corners=True),
                            nn.Conv2d(embed_dims, embed_dims, 
                                    3, padding=1, bias=False),
                            nn.BatchNorm2d(embed_dims),
                            nn.ReLU(inplace=True)))

        self.up = nn.Sequential(*layers)

    def forward(self, x):
        '''
        Input: b n c h w
        Ourput: b c 128 128
        '''

        # generate query
        b, _, c, _, _ = x.shape
        len_q = self.q_generator.query.shape[0]

        q = self.q_generator(x.device)                     # (16*16) 64
        q = q.expand(b, len_q, c)                  # b (16*16) 64


        # TODO k PE
        kv = x.permute(0, 1, 3, 4, 2).reshape(b, -1, c) # b (n h w) c
        # kv = x.reshape(b, c, -1).permute(0,2,1)    # b (200*112) 64


        # matching
        q = self.match(q, kv)                      # b (16*16) 64


        # upsample
        q = rearrange(q, 'b (h w) c -> b c h w', 
                        h=self.q_size, w=self.q_size)   # b 64 16 16
        return self.up(q)                               # b 64 128 128



class QueryGenerator(torch.nn.Module):

    def __init__(self, q_size=16, embed_dims=64):
        super().__init__()

        self.embed_dims = embed_dims
        x = (torch.arange(q_size) + 0.5) / q_size
        y = (torch.arange(q_size) + 0.5) / q_size
        xy=torch.meshgrid(x,y)
        self.query =torch.cat([xy[0].reshape(-1)[...,None],
                                xy[1].reshape(-1)[...,None]],-1)    # (16*16) 2

        self.query_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

    def queryPE(self, query, temperature=10000):
        scale = 2 * math.pi
        query = query * scale   # 256, 64
        dim_t = torch.arange(self.embed_dims/2, 
                            dtype=torch.float32, 
                            device=query.device)

        dim_t = temperature ** (2 * (dim_t / 2) / self.embed_dims/2)

        pos_x = query[..., 0, None] / dim_t
        pos_y = query[..., 1, None] / dim_t
        
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        query = torch.cat((pos_y, pos_x), dim=-1)
        return query


    def forward(self, device):
        q = self.queryPE(self.query.to(device))
        q = self.query_embedding(q)
        return q

