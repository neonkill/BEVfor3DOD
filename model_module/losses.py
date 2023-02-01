# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------


import torch
import logging

import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss


logger = logging.getLogger(__name__)