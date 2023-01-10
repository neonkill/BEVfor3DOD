import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange



class CrossAttention(nn.Module):
    def __init__(self, dim, heads, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        dim_head = int(dim/heads)
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v):
        """
        q: (b d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b d H W -> b (H W) d')
        k = rearrange(k, 'b n d h w -> b (n h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)   # (b m) (H W) d
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)   # (b m) (n h w) d
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)   # (b m) (n h w) d

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b q d, b k d -> b q k', q, k)
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b q k, b k d -> b q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


class TransformerDecoderLayer(torch.nn.Module):

    # __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None):

        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu    #_get_activation_fn(activation)

    # def __setstate__(self, state):
    #     if 'activation' not in state:
    #         state['activation'] = F.relu
    #     super(TransformerDecoderLayer, self).__setstate__(state)

    
    def forward(self, query: Tensor, key: Tensor, value: Tensor):

        query2 = self.self_attn(query, query, query)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.multihead_attn(query, key, value)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query