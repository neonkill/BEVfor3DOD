# ----------------
# Author: yelin2
# ----------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from model_module.model.segmentation.matching import CrossAttention
'''

'''

class SemanticFeatureExtractor(torch.nn.Module):

    def __init__(self, bev_embedding, image_height, image_width, 
                b_res=16, embed_dims=64, n_head=4, q_size=16, full_q_size=128):
        super().__init__()

        self.q_size = q_size
        self.embed_dims = embed_dims

        # for query positional embedding
        self.bev_embedding = BEVEmbedding(**bev_embedding)
        self.bev_embed = nn.Conv2d(2, embed_dims, 1)



        self.match = CrossAttention(dim=embed_dims, 
                                    heads=n_head, 
                                    qkv_bias=True)

        
        # for key positional embedding
        feat_height = math.ceil(image_height/b_res)
        feat_width = math.ceil(image_width/b_res)

        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.img_embed = nn.Conv2d(4, embed_dims, 1, bias=False)
        self.t_embed = nn.Conv2d(4, embed_dims, 1, bias=False)

        self.val_linear = nn.Sequential(
                            # nn.BatchNorm2d(embed_dims),
                            # nn.ReLU(),
                            nn.Conv2d(embed_dims, embed_dims, 1, bias=False))
        self.key_linear = nn.Sequential(
                            # nn.BatchNorm2d(embed_dims),
                            # nn.ReLU(),
                            nn.Conv2d(embed_dims, embed_dims, 1, bias=False))

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

    def forward(self, x, I, E):
        '''
        Input: b n c h w
        Ourput: b c 128 128
        '''
        
        b, n, c, h, w = x.shape
        I_inv = I.inverse()
        E_inv = E.inverse()

        # Generate value
        img_flat = rearrange(x, 'b n ... -> (b n) ...')                  # (b n) d H W
        val_flat = self.val_linear(img_flat)                                    # (b n) d H W
        VAL = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d H W


        # generate query
        Xw = self.bev_embedding.grid[:2]                                    # 2 BH BW (= 2 16 16)
        query = self.bev_embed(Xw[None])                                    # 1 d BH BW
        QUERY = query.expand(b, self.embed_dims, self.q_size, self.q_size)  # b d BH BW
        # query = rearrange(query, 'b d bh bw -> b (bh bw) d')                # b (BH BW) d



        # generate key positional embedding
        pixel = self.image_plane                                                # 2 H W
        _, _, _, h, w = pixel.shape

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (H W)
        cam = I_inv @ pixel_flat                                                # b n 3 (H W)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (H W)
        d = E_inv @ cam                                                         # b n 4 (H W)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 H W
        d_embed = self.img_embed(d_flat)                                        # (b n) d H W

        T = E_inv[...,-1:]                                                      # b n 4 1
        T_flat = rearrange(T, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        T_embed = self.t_embed(T_flat)                                          # (b n) d 1 1

        img_embed = d_embed - T_embed                                           # (b n) d H W
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W

        key_flat = img_embed + self.key_linear(img_flat)
        KEY = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d H W


        BEV = self.match(QUERY, KEY, VAL)                                       # b d H W

        return self.up(BEV)                                                   # b 64 128 128


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


# def generate_grid(height: int, width: int):
#     xs = torch.linspace(0, 1, width)
#     ys = torch.linspace(0, 1, height)

#     indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
#     indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
#     indices = indices[None]                                                 # 1 3 h w

#     return indices


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