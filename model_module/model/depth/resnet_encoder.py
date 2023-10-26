import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
try:
    from model_module.model.modules import *
except:
    import sys
    sys.path.append('../')
    from modules import *

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    def __init__(self, layers = 18, pretrained = False, num_input_images = 1):
        super(ResnetEncoder, self).__init__()
        self.num_enc_ch = np.array([64, 64, 128, 256, 512])

        resnets = {18 : models.resnet18,
                  34 : models.resnet34,
                  50 : models.resnet50,
                  101: models.resnet101,
                  152: models.resnet152}

        if layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers.".format(layers))
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers = layers, pretrained = pretrained, num_input_images=num_input_images)
        else :
            self.encoder = resnets[layers]()

        if layers > 34:
            self.num_enc_ch[1:] *= 4
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        
        # self.feats4_conv = ConvBlock(96, 64, 1, 1) #그냥 conv만 쓸지도?
        # self.merge_4 = Merging(in_channels=64, out_channels=128, downscaling_factor=2) #1/4 -> 1/8
        # self.merge_8 = Merging(in_channels=128, out_channels=256, downscaling_factor=2) #1/8 -> 1/16
        # self.merge_16 = Merging(in_channels=256, out_channels=512, downscaling_factor=2) #1/16 -> 1/32
        
        
    def forward(self, inputs):
        # x = (inputs - 0.45) / 0.225 Monodepth2 (rough approximation of the imagenet pretraining. Need to experiment)
        #https://github.com/nianticlabs/monodepth2/issues/12
        #* training feature shape : 3, 192, 640
        features = []
        # seg_embedding4 = self.feats4_conv(agg4_feats)
        # seg_embedding8 = self.merge_4(seg_embedding4)
        # seg_embedding16 = self.merge_8(seg_embedding8)
        # seg_embedding32 = self.merge_16(seg_embedding16)
        
        x = self.encoder.conv1(inputs)
        x = self.encoder.bn1(x)
        x2 = self.encoder.relu(x) #self.num_enc_ch[0] x H/2 x W/2
        # features.append(x2)
        x4 = self.encoder.maxpool(x2)
        # x4 = x4 + seg_embedding4
        x4 = self.encoder.layer1(x4) #self.num_enc_ch[1] x H/4 x W/4
        features.append(x4)
        x8 = self.encoder.layer2[0](x4) #self.num_enc_ch[2] x H/8 x W/8
        # x8 = x8 + seg_embedding8
        x8 = self.encoder.layer2[1](x8) #self.num_enc_ch[2] x H/8 x W/8
        features.append(x8)
        
        x16 = self.encoder.layer3[0](x8) #self.num_enc_ch[3] x H/16 x W/16
        # x16 = x16 + seg_embedding16
        x16 = self.encoder.layer3[1](x16) #self.num_enc_ch[3] x H/16 x W/16
        features.append(x16)
        
        x32 = self.encoder.layer4[0](x16) #self.num_enc_ch[4] x H/32 x W/32
        # x32 = x32 + seg_embedding32
        x32 = self.encoder.layer4[1](x32) #self.num_enc_ch[4] x H/32 x W/32        
        x32 = self.encoder.avgpool(x32) #identity
        x32 = self.encoder.fc(x32) #identity
        features.append(x32)
        

        return features

