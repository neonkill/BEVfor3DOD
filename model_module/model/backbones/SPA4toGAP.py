# ----------------
# Author: yelin2
# ----------------

import torch
import torch.nn as nn

from model_module.model.modules import Aggregator



class SPA4toGAP(torch.nn.Module):

    def __init__(self, efficientnet, reduce_dim, chs):
        super().__init__()

        self.efficientnet = efficientnet

        self.aggregator = Aggregator(in_dim=reduce_dim,
                                    bins=chs)

        

    def forward(self, x):
        feats = self.efficientnet(x)    # 4, 8, 16, 32, 64, GAP
        return self.aggregator(feats[0], feats[1:])