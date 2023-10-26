import math
from tkinter.tix import Tree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from ..modules import LayerNorm, ConvBlock


class Matching(nn.Module):
    def __init__(
        self,
        cross_attn,
        bev_embedding: dict,
        image_height: int,
        image_width: int,
        dim = 128,
        heads = 4,
        b_res = 32,
        is_full = True,
        norm = 'BN'
    ):
        super().__init__()

        if norm == 'BN':
            normalization = nn.BatchNorm2d
        elif norm == 'LN':
            normalization = LayerNorm
        self.is_full = is_full
        self.bev_embedding = BEVEmbedding(**bev_embedding)


        if (image_height == 320) or (image_height == 448):
            feat_height = math.ceil(image_height/b_res)
            feat_width = math.ceil(image_width/b_res)
        elif (image_height == 450) or (image_height == 900):
            feat_height = round(image_height/b_res)#+1
            feat_width = round(image_width/b_res)
        else:
            AssertionError('feat_height & feat_width is not define')


        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)


        self.t_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)

        if norm == 'LN':
            self.val_linear = nn.Sequential(
                                normalization(dim, eps=1e-6, data_format = "channels_first"),
                                nn.GELU(),
                                nn.Conv2d(dim, dim, 1, bias=False))
            self.key_linear = nn.Sequential(
                                normalization(dim, eps=1e-6, data_format = "channels_first"),
                                nn.GELU(),
                                nn.Conv2d(dim, dim, 1, bias=False))
        elif norm == 'BN':
            self.val_linear = nn.Sequential(
                                normalization(dim),
                                nn.ReLU(),
                                nn.Conv2d(dim, dim, 1, bias=False))
            self.key_linear = nn.Sequential(
                                normalization(dim),
                                nn.ReLU(),
                                nn.Conv2d(dim, dim, 1, bias=False))

        assert dim % heads == 0, 'number of multi attention heads is wrong..'
        self.cross_attn = cross_attn

        # if is_full:
        #     layers = []
        #     bev_resolution = bev_embedding['resolution']
        #     # bev_resolution = 4
            
        #     dim = 384
            
        #     if norm == 'LN':
        #         for _ in range(int(math.log2(bev_resolution))):
                    
        #             layers.append(nn.Sequential(
        #                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                             nn.Conv2d(dim, dim//2, 3, padding=1, bias=False),
        #                             normalization(dim//2, eps=1e-6, data_format = "channels_first"),
        #                             nn.GELU()))
        #             dim = dim//2
                    
        #     elif norm == 'BN':
        #         for _ in range(int(math.log2(bev_resolution))):
        #             layers.append(nn.Sequential(
        #                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #                             nn.Conv2d(dim, dim//2, 3, padding=1, bias=False),
        #                             normalization(dim//2),
        #                             nn.ReLU(inplace=True)))
        #             dim = dim//2
                    
        #     self.up = nn.Sequential(*layers)


    def forward(self, img_feat, I, E):
        '''
        generate deformable transformer's query using transformer
        Input:
            img_feat: 1/128 feature from image (b, 6, 128, h/128, w/128)
            I: intrinsic matrix (b, n, 3, 3)
            E: extrinsic matrix (b, n, 4, 4)
        Output:
            query for deformable transformer (b, 128, 200, 200)
        '''

        b, n, _, _, _ = img_feat.shape

        I_inv = I.inverse()
        E_inv = E.inverse()

        # Generate tau_k (translation embedding feature)
        T = E_inv[...,-1:]                                          # b n 4 1
        T_flat = rearrange(T, 'b n ... -> (b n) ...')[..., None]    # (b n) 4 1 1
        T_embed = self.t_embed(T_flat)                              # (b n) d 1 1

        # Generate query
        Xw = self.bev_embedding.grid[:2]                                        # 2 BH BW (= 2 50 50)
        bev_embed = self.bev_embed(Xw[None])                                    # 1 d BH BW
        bev_embed = bev_embed - T_embed                                         # (b n) d BH BW
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    
        QUERY = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)          # b n d BH BW

        # Generate value
        img_flat = rearrange(img_feat, 'b n ... -> (b n) ...')                  # (b n) d H W
        val_flat = self.val_linear(img_flat)                                    # (b n) d H W
        VAL = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d H W

        # Generate key
        pixel = self.image_plane                                                # 2 H W (2 7 13)
        _, _, _, h, w = pixel.shape

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (H W)
        cam = I_inv @ pixel_flat                                                # b n 3 (H W)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (H W)
        d = E_inv @ cam                                                         # b n 4 (H W)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 H W
        d_embed = self.img_embed(d_flat)                                        # (b n) d H W
        
        img_embed = d_embed - T_embed                                           # (b n) d H W
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        
        # print(img_embed.shape, img_flat.shape)
        key_flat = img_embed + self.key_linear(img_flat)
        
        KEY = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d H W


        # Cross Attention
        BEV4 = self.cross_attn(QUERY, KEY, VAL)
        # print('bev4', BEV4.shape)

        # Upsample BEV to full resolution
        # BEV = self.up(BEV4) if self.is_full else BEV4

        # print('bev', BEV.shape)
        

        return BEV4


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


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        dim_head = int(dim/heads)
        self.scale = dim_head ** -0.5

        self.heads = heads # 4
        self.dim_head = dim_head # 16

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
        
        self.conv1x1 = ConvBlock(dim * 6 , 256, 1, norm='LN', act='GELU')

    def forward(self, q, k, v):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        attention per cam !! !
        
        out shape : B, 6 * dim, q_H, q_W (6 : number of cam)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b n (h w) d') #! for seprate
        # v = rearrange(v, 'b n d h w -> b (n h w) d') #! original

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n q d, b n k d -> b n q k', q, k)
        # dot = rearrange(dot, 'b n q k -> b q (n k)') #!original
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        # a = torch.einsum('b q k, b k d -> b q d', att, v) #!original
        a = torch.einsum('b n q k, b n k d -> b n q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        # z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)
        z = rearrange(z, 'b n (H W) d -> b n d H W', H=H, W=W)
        
        #! for sep
        # z = torch.sum(z, dim=1)
        z = rearrange(z, 'b n d H W -> b (n d) H W') # concat for cam
        
        z = self.conv1x1(z) #shake feats
        
        return z