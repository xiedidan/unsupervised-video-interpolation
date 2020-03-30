# ==============================================================================
# Copyright 2019 Florent Mahoudeau.
# Modified from https://github.com/juntang-zhuang/ShelfNet
# Licensed under the MIT License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shelf_base import BaseNet

__all__ = ['ShelfNet', 'get_shelf_unet']

class ShelfNet(BaseNet):
    def __init__(self, n_joints, backbone, extra_block, tail_fuse_method='pyramid',
                 norm_layer=None, dilated=False, **kwargs):
        super(ShelfNet, self).__init__(n_joints, backbone, norm_layer=norm_layer,
                                       dilated=dilated, **kwargs)

        self.extra_block = extra_block
        self.tail_fuse_method = tail_fuse_method
        self.head = ShelfHead(self.extra_block, self.tail_fuse_method)

        if self.tail_fuse_method.lower() == 'pyramid':
            self.final = nn.Conv2d(64, n_joints, 1)
        elif self.tail_fuse_method.lower() == 'concat':
            self.final = nn.Conv2d(64+128+256+512, n_joints, 1)
        else:
            raise ValueError('Unknown fuse method: {}'.format(self.tail_fuse_method))

    def forward(self, x):
        features = self.base_forward(x)
        x0, out = self.head(features)
        pred = self.final(out[-1])

        return x0, pred

class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()

        self.planes = planes
        self.conv1 = nn.Conv2d(self.planes, self.planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = out + x

        return out


class ShelfPyramidDecoder(nn.Module):
    def __init__(self, block=BasicBlock):
        super().__init__()

        # create basic blocks
        self.block_a = block(64)
        self.block_b = block(128)
        self.block_c = block(256)
        self.block_d = block(512)

        # create up-sampling layers
        self.up_conv_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        self.up_conv_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1)
        )
        self.up_conv_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1)
        )

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.block_d(x4)
        
        out2 = self.up_conv_1(out1) + x3
        out2 = self.block_c(out2)

        out3 = self.up_conv_2(out2) + x2
        out3 = self.block_b(out3)

        out4 = self.up_conv_3(out3) + x1
        out4 = self.block_a(out4)

        return [out1, out2, out3, out4]


class ShelfConcatDecoder(nn.Module):
    def __init__(self, block=BasicBlock):
        super().__init__()

        # create basic blocks
        self.block_a = block(64)
        self.block_b = block(128)
        self.block_c = block(256)
        self.block_d = block(512)

        # create up-sampling layers
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=64+128+256+512,
                out_channels=64+128+256+512,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.block_d(x4)
        out2 = self.block_c(x3)
        out3 = self.block_b(x2)
        out4 = self.block_a(x1)

        # Head part
        height, width = x1.size(2), x1.size(3)
        out1 = F.interpolate(out1, size=(height, width), mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=(height, width), mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=(height, width), mode='bilinear', align_corners=False)
        out = torch.cat([out1, out2, out3, out4], 1)
        out = self.head(out)

        return [out]


class ShelfHead(nn.Module):
    def __init__(self, extra_block, tail_fuse_method='pyramid'):
        super(ShelfHead, self).__init__()

        self.decoder = ShelfPyramidDecoder()
        self.tail = ShelfTail(extra_block, tail_fuse_method)

    def forward(self, x):
        x0, x1, x2, x3, x4 = x

        out = self.decoder([x1, x2, x3, x4])
        out = self.tail(out)  # Second encoder-decoder pass

        return x0, out


class ShelfTail(nn.Module):
    def __init__(self, extra_block, fuse_method, block=BasicBlock):
        super().__init__()

        # create basic blocks
        self.extra_block = extra_block
        if self.extra_block:
            self.block_in = block(64)
        self.block_a = block(64)
        self.block_b = block(128)
        self.block_c = block(256)

        # create down-sampling layers
        self.down_conv_1 = nn.Conv2d(64, 128, stride=2, kernel_size=3, padding=1)
        self.down_conv_2 = nn.Conv2d(128, 256, stride=2, kernel_size=3, padding=1)
        self.down_conv_3 = nn.Conv2d(256, 512, stride=2, kernel_size=3, padding=1)

        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fuse_method = fuse_method
        if self.fuse_method.lower() == 'pyramid':
            self.decoder = ShelfPyramidDecoder()
        elif self.fuse_method.lower() == 'concat':
            self.decoder = ShelfConcatDecoder()
        else:
            raise ValueError('Unknown fuse method: {}'.format(self.fuse_method))

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.block_in(x4) + x4 if self.extra_block else x4
        out1 = self.block_a(out1)

        out2 = self.down_conv_1(out1)
        out2 = self.relu1(out2) + x3
        out2 = self.block_b(out2)

        out3 = self.down_conv_2(out2)
        out3 = self.relu2(out3) + x2
        out3 = self.block_c(out3)

        out4 = self.down_conv_3(out3)
        out4 = self.relu3(out4)

        out = self.decoder([out1, out2, out3, out4])

        return out

def get_shelf_unet(**kwargs):
    model = ShelfNet(
        n_joints=17,
        backbone='resnet18',
        extra_block=False,
        tail_fuse_method='pyramid',
        **kwargs
    )
    
    return model
