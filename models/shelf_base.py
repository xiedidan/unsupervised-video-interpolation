# ==============================================================================
# Copyright 2019 Florent Mahoudeau.
# Modified from https://github.com/juntang-zhuang/ShelfNet
# Licensed under the MIT License.
# ==============================================================================

import torch.nn as nn
import sys

from .resnet_bnfree_lrelu import *
from .model_utils import *

__all__ = ['BaseNet']

class BaseNet(nn.Module):
    def __init__(self, n_joints, backbone, dilated=True, norm_layer=None,
                 base_size=576, crop_size=608, mean=[.485, .456, .406],
                 std=[.229, .224, .225], input_channel=6):
        super(BaseNet, self).__init__()
        self.n_joints = n_joints
        self.backbone = backbone
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        
        self.conv1 = nn.Sequential(
            Factorized_Conv3d(input_channel, 4, stride=2),
            Factorized_Conv3d(4, 8, stride=2),
            Factorized_Conv3d(8, 16, stride=2)
        )

        # copying modules from pretrained models
        if backbone == 'resnet18':
            self.pretrained = resnet18(
                pretrained=True,
                norm_layer=norm_layer,
                input_channel=input_channel,
                first_stride=4,
                first_lite=True,
                inplanes=16,
                stride=2
            )
        elif backbone == 'resnet34':
            self.pretrained = resnet34(
                pretrained=True,
                norm_layer=norm_layer,
                input_channel=input_channel,
                first_stride=4,
                first_lite=True,
                inplanes=16,
                stride=2
            )
        else:
            raise RuntimeError('Unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        x = self.conv1(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        
        return x, c1, c2, c3, c4