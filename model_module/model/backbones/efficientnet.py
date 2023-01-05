# -----------------------------------------------------------------------
# Copyright (C) 2020 Brady Zhou
# Released under the MIT license
# https://github.com/bradyz/cross_view_transformers/blob/master/LICENSE
# -----------------------------------------------------------------------
# Modified by yelin2
# -----------------------------------------------------------------------


import torch
import torch.utils.checkpoint
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

# 다 내려가는 모델
MODELS = {
    'efficientnet-b0': [
        ('reduction_1', (0, 3)),
        ('reduction_2', (3, 5)),
        ('reduction_3', (5, 11)),
        ('reduction_4', (11, 15)),
        ('reduction_5', (15, 17))
    ],

    'efficientnet-b2': [
        ('reduction_1', (0, 5)),
        ('reduction_2', (5, 8)),
        ('reduction_3', (8, 16)),
        ('reduction_4', (16, 21)),
        ('reduction_5', (21, 23))
    ],
    'efficientnet-b4': [
        ('reduction_1', (0, 6)),    # 2
        ('reduction_2', (6, 10)),    # 4
        ('reduction_3', (10, 22)),   # 8
        ('reduction_4', (22, 32)),  # 16
        ('reduction_5', (32, 33))   # 32
    ],

}


class EfficientNetExtractor(torch.nn.Module):
    """
    Helper wrapper that uses torch.utils.checkpoint.checkpoint to save memory while training.

    This runs a fake input with shape (1, 3, input_height, input_width)
    to give the shapes of the features requested.

    Sample usage:
        backbone = EfficientNetExtractor(224, 480, ['reduction_2', 'reduction_4'])

        # [[1, 56, 28, 60], [1, 272, 7, 15]]
        backbone.output_shapes

        # [f1, f2], where f1 is 'reduction_1', which is shape [b, d, 128, 128]
        backbone(x)
    """
    def __init__(self, layers, extra_layers, chs, reduce_dim, image_height, image_width, model_name='efficientnet-b4'):
        super().__init__()

        assert model_name in MODELS
        assert all(k in [k for k, v in MODELS[model_name]] for k in layers)

        idx_max = -1
        layer_to_idx = {}

        # Find which blocks to return
        for i, (layer_name, _) in enumerate(MODELS[model_name]):
            if layer_name in layers:
                idx_max = max(idx_max, i)
                layer_to_idx[layer_name] = i

        length = max(layer_to_idx.values())
        for i, layer in enumerate(extra_layers):
            layer_to_idx[layer] = length + i + 1

        # We can set memory efficient swish to false since we're using checkpointing
        net = EfficientNet.from_pretrained(model_name)
        net.set_swish(False)

        drop = net._global_params.drop_connect_rate / len(net._blocks)
        blocks = [nn.Sequential(net._conv_stem, net._bn0, net._swish)]

        # Only run needed blocks
        for idx in range(idx_max):
            l, r = MODELS[model_name][idx][1]

            block = SequentialWithArgs(*[(net._blocks[i], [i * drop]) for i in range(l, r)])
            blocks.append(block)

        for i, layer in enumerate(extra_layers):
            idx = (i) + len(layers)
            if layer == 'GAP':
                blocks.append(nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(chs[idx], chs[idx], 
                                    kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(chs[idx]),
                            nn.ReLU(inplace=True)))
            else:
                blocks.append(nn.Sequential(
                            nn.Conv2d(chs[idx], chs[idx], 
                                    kernel_size=3, stride=2,
                                    padding=1, bias=False),
                            nn.BatchNorm2d(chs[idx]),
                            nn.ReLU(inplace=True)))
                            
        self.layers = nn.Sequential(*blocks)
        self.layer_names = layers + extra_layers
        self.idx_pick = [layer_to_idx[l] for l in self.layer_names]

        red_layers = []
        for channel in chs:
            red_layers.append(nn.Sequential(
                                    nn.Conv2d(channel, reduce_dim, 
                                            kernel_size=1, stride=1, bias=False),
                                    # nn.BatchNorm2d(reduce_dim),
                                    # nn.ReLU(inplace=True)
                                    ))
        self.red_layers = nn.Sequential(*red_layers)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # # Pass a dummy tensor to precompute intermediate shapes
        # dummy = torch.rand(2, 3, image_height, image_width)
        # output_shapes = [x.shape for x in self(dummy)]

        # self.output_shapes = output_shapes

        

    def forward(self, x):
        if self.training:
            x = x.requires_grad_(True)

        result = []

        for i, layer in enumerate(self.layers):
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            # print(x.shape)
            result.append(x)

        return [self.red_layers[i](result[idx]) for i, idx in enumerate(self.idx_pick)]


class SequentialWithArgs(nn.Sequential):
    def __init__(self, *layers_args):
        layers = [layer for layer, args in layers_args]
        args = [args for layer, args in layers_args]

        super().__init__(*layers)

        self.args = args

    def forward(self, x):
        for l, a in zip(self, self.args):
            x = l(x, *a)

        return x