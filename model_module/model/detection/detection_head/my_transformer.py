# ----------------
# Author: neonkill
# ----------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.builder import TRANSFORMER
import copy

@TRANSFORMER.register_module()
class Customtransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, num_decoder_layers=6, feedforward_channels=256,
                dropout=0.1, return_intermediate=False):
        super().__init__()

        self.decoder_layer = CustomTransformerDecoderLayer(embed_dim,num_heads,feedforward_channels,dropout)
        self.decoder = CustomTransformerDecoder(self.decoder_layer, num_decoder_layers, return_intermediate=return_intermediate)
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, query_pos, key, value, key_pos=None):
        batch_size, _, _, _ = key.shape

        key = key.flatten(2).permute(2,0,1)
        value = value.flatten(2).permute(2,0,1)# b c h w -> hw b c
        query_pos = query_pos.unsqueeze(1).repeat(1, batch_size, 1)# hw c -> hw b c
        # query_pos = query_pos.flatten(2).permute(2,0,1).repeat(1, batch_size, 1)
        if key_pos is not None:
            key_pos = key_pos.flatten(2).permute(2,0,1)
            # key_pos = key_pos.unsqueeze(1).repeat(1, batch_size, 1)
        # query = query.flatten(2).permute(2,0,1) # b c h w -> hw b c
        query = query.unsqueeze(1).repeat(1, batch_size, 1)

        
        output = self.decoder(query=query, key=key, value=value, query_pos=query_pos, key_pos=key_pos)

        return output.transpose(1,2)


class CustomTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_decoder_layers, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_decoder_layers)])
        self.return_intermediate = return_intermediate

    def forward(self, query, key, value, query_pos, key_pos=None):

        output = query

        if self.return_intermediate:
            intermediate = []

        for layer in self.layers:
            output = layer(output, key, value, query_pos, key_pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_channels, dropout):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, feedforward_channels)
        self.linear2 = nn.Linear(feedforward_channels, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = F.gelu

    def pos_embed(self, tensor, pos):
        if pos is not None:
            return tensor + pos
        return tensor

    def forward(self, query, key, value, query_pos, key_pos=None):
        q = k = self.pos_embed(query, query_pos)
        
        # query_attn = self.self_attention(query, key, value)[0]
        query_attn = self.self_attention(q, k, query)[0]

        query = query + self.dropout1(query_attn)
        query = self.norm1(query)
        
        query = self.pos_embed(query, query_pos)
        key = self.pos_embed(key, key_pos)
        
        query_attn1 = self.cross_attention(query, key, value)[0]
        query = query + self.dropout2(query_attn1)
        query = self.norm2(query)

        query_ffn = self.linear2(self.dropout3(self.activation(self.linear1(query))))
        query = query + self.dropout4(query_ffn)
        query = self.norm3(query)

        return query

# class CustomTransformerDecoderLayer(nn.Module):
#     def __init__(self, embed_dim, num_heads, feedforward_channels, dropout):
#         super().__init__()
#         self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

#         self.linear1 = nn.Linear(embed_dim, feedforward_channels)
#         self.linear2 = nn.Linear(feedforward_channels, embed_dim)

#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = F.gelu

#     def pos_embed(self, tensor, pos=None):
#         if pos is not None:
#             return tensor + pos
#         return tensor

#     def forward(self, query, key, value, query_pos, key_pos=None):
#         q = self.pos_embed(query, query_pos)
#         if key_pos is None:
#             key_pos = query_pos
#         k = self.pos_embed(key, key_pos)
        
#         query_attn = self.self_attention(q, k, value)[0]

#         query = query + self.dropout1(query_attn)
#         query = self.norm1(query)

#         query_ffn = self.linear2(self.dropout2(self.activation(self.linear1(query))))
#         query = query + self.dropout3(query_ffn)
#         query = self.norm2(query)

#         return query
