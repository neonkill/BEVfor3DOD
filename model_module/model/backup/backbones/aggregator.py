# ----------------
# Author: yelin2
# ----------------

import torch
import torch.nn as nn

from model_module.model.modules import Aggregator



class Agg4andMerge16(torch.nn.Module):

    def __init__(self, efficientnet, reduce_dim, chs):
        super().__init__()

        self.efficientnet = efficientnet

        self.aggregator = Aggregator(in_dim=reduce_dim,
                                    bins=chs)

        

    def forward(self, x):
        feats = self.efficientnet(x)    # 4, 8, 16, 32, 64, GAP
        agg = self.aggregator(feats[0], feats[1:])
        # TODO 1/16 feature 만드는거

        return 


class Agg16Agg4(torch.nn.Module):

    def __init__(self, efficientnet, reduce_dim, chs):
        super().__init__()
        
        # reduce_dim = 64
        # chs = [32, 56, 160, 448, 448, 448]

        self.efficientnet = efficientnet

        self.aggregator16 = Aggregator(in_dim=reduce_dim,
                                    bins=chs)
        # self.aggregator4 = Aggregator(in_dim=reduce_dim,
        #                             bins=chs)

        # self.aggregator16 = Aggregator(in_dim=reduce_dim,
        #                             bins=chs[2:])
        

    def forward(self, x):
        
        feats = self.efficientnet(x)    # 4, 8, 16, 32, 64, GAP

        agg16 = self.aggregator16(feats[0], feats[1:])   # B 64 h/16 w/16
        
        return agg16
        # agg16 = self.aggregator16(feats[2], feats[3:])   # B 64 h/16 w/16
        # agg4 = self.aggregator4(feats[0], feats[1:])    # B 64 h/4 w/4
        
        # return agg4, agg16


class BB32(torch.nn.Module):

    def __init__(self, efficientnet):
        super().__init__()
        
        # reduce_dim = 64
        # chs = [32, 56, 160, 448, 448, 448]

        self.efficientnet = efficientnet
        

    def forward(self, x):
        
        feats = self.efficientnet(x)    # 4, 8, 16, 32, 64, GAP
        
        return feats[0]
