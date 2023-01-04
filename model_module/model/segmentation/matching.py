import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor



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
    
    
    # def forward(self, tgt: Tensor, memory: Tensor)

    #     tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt = tgt + self.dropout1(tgt2)
    #     tgt = self.norm1(tgt)
    #     tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
    #                                key_padding_mask=memory_key_padding_mask)[0]
    #     tgt = tgt + self.dropout2(tgt2)
    #     tgt = self.norm2(tgt)
    #     tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    #     tgt = tgt + self.dropout3(tgt2)
    #     tgt = self.norm3(tgt)
    #     return tgt