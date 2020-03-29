# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .model_utils import MyResample2D

class FactoriedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1):
        super(FactoriedConv2d, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, (kernel_size, 1), padding=(padding, 0))
        self.conv2 = nn.Conv2d(output_channels, output_channels, (1, kernel_size), padding=(0, padding))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()

        self.conv = nn.Conv2d(out_channels, 1, 1, padding=0)

    def forward(self, x):
        x=self.conv(x)
        x=torch.sigmoid(x)

        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()

        self.conv1 = nn.Conv2d(out_channels, int(out_channels/2), 1, padding=0)
        self.conv2 = nn.Conv2d(int(out_channels/2), out_channels, 1, padding=0)

    def forward(self, x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        x=self.conv1(x)
        x=F.leaky_relu(x, negative_slope=0.1)
        x=self.conv2(x)
        x=torch.sigmoid(x)

        return x

class Decoder(nn.Module):
    def __init__(self, in_chn, out_chn, scse=False, factorize=False):
        super(Decoder, self).__init__()

        self.scse = scse
        self.factorize = factorize

        conv = FactoriedConv2d if self.factorize else nn.Conv2d

        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv(in_chn, out_chn, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            conv(out_chn, out_chn, 3, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

        if self.scse:
            self.spatial_gate = sSE(out_chn)
            self.channel_gate = cSE(out_chn)

    def forward(self, x):
        x = self.layer(x)

        if self.scse:
            g1 = self.spatial_gate(x)
            g2 = self.channel_gate(x)

            x = g1 * x + g2 * x

        return x
    
class HJSuperSloMoMod(nn.Module):
    def __init__(self, args, mean_pix=[109.93, 109.167, 101.455], in_channel=6):
        super(HJSuperSloMoMod, self).__init__()

        self.is_output_flow = False
        
        self.hc = args.hc
        self.lite_encoder = args.lite_encoder
        self.scse = args.scse
        self.backbone = args.backbone
        self.factorize = args.factorize
        
        # --------------------- encoder --------------------
        # conv1
        if self.backbone is None:
            self.flow_pred_encoder_layer1 = self.make_flow_pred_encoder_layer(in_channel, 32, 7, 3, lite=self.lite_encoder, factorize=self.factorize)
            self.flow_pred_encoder_layer2 = self.make_flow_pred_encoder_layer(32, 64, 5, 2, lite=self.lite_encoder, factorize=self.factorize)
            self.flow_pred_encoder_layer3 = self.make_flow_pred_encoder_layer(64, 128)
            self.flow_pred_encoder_layer4 = self.make_flow_pred_encoder_layer(128, 256)
            self.flow_pred_encoder_layer5 = self.make_flow_pred_encoder_layer(256, 512)
            
            self.flow_pred_bottleneck = self.make_flow_pred_encoder_layer(512, 512)

            self.flow_pred_decoder_layer5 = self.make_flow_pred_decoder_layer(512, 512, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer4 = self.make_flow_pred_decoder_layer(1024, 256, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer3 = self.make_flow_pred_decoder_layer(512, 128, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer2 = self.make_flow_pred_decoder_layer(256, 64, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer1 = self.make_flow_pred_decoder_layer(128, 32, scse=self.scse, factorize=self.factorize)

            self.flow_pred_refine_layer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
        elif 'resnet' in self.backbone.__name__:
            self.flow_pred_backbone = self.backbone(pretrained=True, progress=True)

            self.flow_pred_encoder_layer1 = self.make_flow_pred_encoder_layer(in_channel, 64, 7, 3, lite=self.lite_encoder, factorize=self.factorize)
            self.flow_pred_encoder_layer2= self.flow_pred_backbone.layer1
            self.flow_pred_encoder_layer3= self.flow_pred_backbone.layer2
            self.flow_pred_encoder_layer4= self.flow_pred_backbone.layer3
            self.flow_pred_encoder_layer5 = self.flow_pred_backbone.layer4
            
            self.flow_pred_bottleneck = self.make_flow_pred_encoder_layer(512, 512)

            self.flow_pred_decoder_layer5 = self.make_flow_pred_decoder_layer(512, 512, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer4 = self.make_flow_pred_decoder_layer(1024, 256, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer3 = self.make_flow_pred_decoder_layer(512, 128, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer2 = self.make_flow_pred_decoder_layer(256, 64, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer1 = self.make_flow_pred_decoder_layer(128, 32, scse=self.scse, factorize=self.factorize)

            self.flow_pred_refine_layer = nn.Sequential(
                nn.Conv2d(96, 32, 3, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
        elif 'shufflenet' in self.backbone.__name__:
            self.flow_pred_backbone = self.backbone(pretrained=True, progress=True)

            self.flow_pred_encoder_layer1 = self.make_flow_pred_encoder_layer(in_channel, 24, 7, 3, lite=self.lite_encoder, factorize=self.factorize) # 24, 1/2
            self.flow_pred_encoder_layer2 = self.make_flow_pred_encoder_layer(24, 24, 3, 1, lite=self.lite_encoder, factorize=self.factorize) # 24, 1/4
            self.flow_pred_encoder_layer3 = self.flow_pred_backbone.stage2 # 116, 1/8
            self.flow_pred_encoder_layer4 = self.flow_pred_backbone.stage3 # 232, 1/16
            self.flow_pred_encoder_layer5 = self.flow_pred_backbone.stage4 # 464, 1/32
            
            self.flow_pred_bottleneck = self.make_flow_pred_encoder_layer(464, 512)

            self.flow_pred_decoder_layer5 = self.make_flow_pred_decoder_layer(512, 512, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer4 = self.make_flow_pred_decoder_layer(512+464, 256, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer3 = self.make_flow_pred_decoder_layer(256+232, 128, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer2 = self.make_flow_pred_decoder_layer(128+116, 64, scse=self.scse, factorize=self.factorize)
            self.flow_pred_decoder_layer1 = self.make_flow_pred_decoder_layer(64+24, 32, scse=self.scse, factorize=self.factorize)

            self.flow_pred_refine_layer = nn.Sequential(
                nn.Conv2d(32+24, 32, 3, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
        
        if not self.hc:
            self.forward_flow_conv = nn.Conv2d(32, 2, 1)
            self.backward_flow_conv = nn.Conv2d(32, 2, 1)
        else:
            # 256+128+64+32+32
            self.forward_flow_conv = nn.Conv2d(512, 2, 1)
            self.backward_flow_conv = nn.Conv2d(512, 2, 1)

        # -------------- flow interpolation encoder-decoder --------------
        if self.backbone is None:
            self.flow_interp_encoder_layer1 = self.make_flow_interp_encoder_layer(16, 32, 7, 3, lite=self.lite_encoder, factorize=self.factorize)
            self.flow_interp_encoder_layer2 = self.make_flow_interp_encoder_layer(32, 64, 5, 2, lite=self.lite_encoder, factorize=self.factorize)
            self.flow_interp_encoder_layer3 = self.make_flow_interp_encoder_layer(64, 128)
            self.flow_interp_encoder_layer4 = self.make_flow_interp_encoder_layer(128, 256)
            self.flow_interp_encoder_layer5 = self.make_flow_interp_encoder_layer(256, 512)
            
            self.flow_interp_bottleneck = self.make_flow_interp_encoder_layer(512, 512)

            self.flow_interp_decoder_layer5 = self.make_flow_interp_decoder_layer(1024, 512, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer4 = self.make_flow_interp_decoder_layer(1024, 256, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer3 = self.make_flow_interp_decoder_layer(512, 128, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer2 = self.make_flow_interp_decoder_layer(256, 64, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer1 = self.make_flow_interp_decoder_layer(128, 32, scse=self.scse, factorize=self.factorize)

            self.flow_interp_refine_layer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
        elif 'resnet' in self.backbone.__name__:
            self.flow_interp_backbone = self.backbone(pretrained=True, progress=True)

            self.flow_interp_encoder_layer1 = self.make_flow_interp_encoder_layer(16, 64, 7, 3, lite=self.lite_encoder, factorize=self.factorize)
            self.flow_interp_encoder_layer2 = self.flow_interp_backbone.layer1
            self.flow_interp_encoder_layer3 = self.flow_interp_backbone.layer2
            self.flow_interp_encoder_layer4 = self.flow_interp_backbone.layer3
            self.flow_interp_encoder_layer5 = self.flow_interp_backbone.layer4

            self.flow_interp_bottleneck = self.make_flow_interp_encoder_layer(512, 512)

            self.flow_interp_decoder_layer5 = self.make_flow_interp_decoder_layer(1024, 512, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer4 = self.make_flow_interp_decoder_layer(1024, 256, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer3 = self.make_flow_interp_decoder_layer(512, 128, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer2 = self.make_flow_interp_decoder_layer(256, 64, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer1 = self.make_flow_interp_decoder_layer(128, 32, scse=self.scse, factorize=self.factorize)

            self.flow_interp_refine_layer = nn.Sequential(
                nn.Conv2d(96, 32, 3, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
        elif 'shufflenet' in self.backbone.__name__:
            self.flow_interp_backbone = self.backbone(pretrained=True, progress=True)

            self.flow_interp_encoder_layer1 = self.make_flow_interp_encoder_layer(16, 24, 7, 3, lite=self.lite_encoder, factorize=self.factorize)
            self.flow_interp_encoder_layer2= self.make_flow_interp_encoder_layer(24, 24, 5, 3, lite=self.lite_encoder, factorize=self.factorize)
            self.flow_interp_encoder_layer3= self.flow_interp_backbone.stage2
            self.flow_interp_encoder_layer4= self.flow_interp_backbone.stage3
            self.flow_interp_encoder_layer5 = self.flow_interp_backbone.stage4

            self.flow_interp_bottleneck = self.make_flow_interp_encoder_layer(464, 512)

            self.flow_interp_decoder_layer5 = self.make_flow_interp_decoder_layer(1024, 512, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer4 = self.make_flow_interp_decoder_layer(512+464, 256, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer3 = self.make_flow_interp_decoder_layer(256+232, 128, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer2 = self.make_flow_interp_decoder_layer(128+116, 64, scse=self.scse, factorize=self.factorize)
            self.flow_interp_decoder_layer1 = self.make_flow_interp_decoder_layer(64+24, 32, scse=self.scse, factorize=self.factorize)

            self.flow_interp_refine_layer = nn.Sequential(
                nn.Conv2d(32+24, 32, 3, padding=1),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
            
        if not self.hc:
            self.flow_interp_forward_out_layer = nn.Conv2d(32, 2, 1)
            self.flow_interp_backward_out_layer = nn.Conv2d(32, 2, 1)
        else:
            self.flow_interp_forward_out_layer = nn.Conv2d(512, 2, 1)
            self.flow_interp_backward_out_layer = nn.Conv2d(512, 2, 1)

        # visibility
        if not self.hc:
            self.flow_interp_vis_layer = nn.Conv2d(32, 1, 1)
        else:
            self.flow_interp_vis_layer = nn.Conv2d(512, 1, 1)

        self.resample2d_train = MyResample2D(args.crop_size[1], args.crop_size[0])

        mean_pix = torch.from_numpy(np.array(mean_pix)).float()
        mean_pix = mean_pix.view(1, 3, 1, 1)
        self.register_buffer('mean_pix', mean_pix)

        self.args = args
        self.scale = args.flow_scale

        self.L1_loss = nn.L1Loss()
        self.L2_loss = nn.MSELoss()
        self.ignore_keys = ['vgg', 'grid_w', 'grid_h', 'tlinespace', 'resample2d_train', 'resample2d']
        self.register_buffer('tlinespace', torch.linspace(0, 1, 2 + args.num_interp).float())

        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_features = nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_features.parameters():
            param.requires_grad = False

        # loss weights
        self.pix_alpha = 0.8
        self.warp_alpha = 0.4 
        self.vgg16_alpha = 0.005
        self.smooth_alpha = 1.

    def make_flow_pred_encoder_layer(self, in_chn, out_chn, kernel_size=3, padding=1, lite=False, factorize=False):
        conv = FactoriedConv2d if factorize else nn.Conv2d

        if lite and (kernel_size > 3):
            if kernel_size == 5:
                layer = nn.Sequential(
                    conv(in_chn, in_chn, 3, padding=1),
                    conv(in_chn, out_chn, 3, padding=1),
                    nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    conv(out_chn, out_chn, 3, padding=1),
                    conv(out_chn, out_chn, 3, padding=1),
                    nn.LeakyReLU(inplace=True, negative_slope=0.1))
            else: # we only create 7x7 even when kernel_size > 7
                layer = nn.Sequential(
                    conv(in_chn, in_chn, 3, padding=1),
                    conv(in_chn, in_chn, 3, padding=1),
                    conv(in_chn, out_chn, 3, padding=1),
                    nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    conv(out_chn, out_chn, 3, padding=1),
                    conv(out_chn, out_chn, 3, padding=1),
                    conv(out_chn, out_chn, 3, padding=1),
                    nn.LeakyReLU(inplace=True, negative_slope=0.1))
        else:
            layer = nn.Sequential(
                conv(in_chn, out_chn, kernel_size, padding=padding),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                conv(out_chn, out_chn, kernel_size, padding=padding),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
            
        return layer

    def make_flow_pred_decoder_layer(self, in_chn, out_chn, scse=False, factorize=False):
        layer = Decoder(in_chn, out_chn, scse, factorize)

        return layer

    def make_flow_interp_encoder_layer(self, in_chn, out_chn, kernel_size=3, padding=1, lite=False, factorize=False):
        conv = FactoriedConv2d if factorize else nn.Conv2d

        if lite and (kernel_size > 3):
            if kernel_size == 5:
                layer = nn.Sequential(
                    conv(in_chn, in_chn, 3, padding=1),
                    conv(in_chn, out_chn, 3, padding=1),
                    nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    conv(out_chn, out_chn, 3, padding=1),
                    conv(out_chn, out_chn, 3, padding=1),
                    nn.LeakyReLU(inplace=True, negative_slope=0.1))
            else: # we only create 7x7 even when kernel_size > 7
                layer = nn.Sequential(
                    conv(in_chn, in_chn, 3, padding=1),
                    conv(in_chn, in_chn, 3, padding=1),
                    conv(in_chn, out_chn, 3, padding=1),
                    nn.LeakyReLU(inplace=True, negative_slope=0.1),
                    conv(out_chn, out_chn, 3, padding=1),
                    conv(out_chn, out_chn, 3, padding=1),
                    conv(out_chn, out_chn, 3, padding=1),
                    nn.LeakyReLU(inplace=True, negative_slope=0.1))
        else:
            layer = nn.Sequential(
                conv(in_chn, out_chn, kernel_size, padding=padding),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
                conv(out_chn, out_chn, kernel_size, padding=padding),
                nn.LeakyReLU(inplace=True, negative_slope=0.1))
            
        return layer

    def make_flow_interp_decoder_layer(self, in_chn, out_chn, scse=False, factorize=False):
        layer = Decoder(in_chn, out_chn, scse, factorize)

        return layer

    def make_flow_interpolation(self, in_data, flow_pred_bottleneck_out):
        if self.backbone is None:
            flow_interp_encoder_out1 = self.flow_interp_encoder_layer1(in_data)
            flow_interp_encoder_out1_pool = F.avg_pool2d(flow_interp_encoder_out1, 2, stride=2)

            flow_interp_encoder_out2 = self.flow_interp_encoder_layer2(flow_interp_encoder_out1_pool)
            flow_interp_encoder_out2_pool = F.avg_pool2d(flow_interp_encoder_out2, 2, stride=2)

            flow_interp_encoder_out3 = self.flow_interp_encoder_layer3(flow_interp_encoder_out2_pool)
            flow_interp_encoder_out3_pool = F.avg_pool2d(flow_interp_encoder_out3, 2, stride=2)

            flow_interp_encoder_out4 = self.flow_interp_encoder_layer4(flow_interp_encoder_out3_pool)
            flow_interp_encoder_out4_pool = F.avg_pool2d(flow_interp_encoder_out4, 2, stride=2)

            flow_interp_encoder_out5 = self.flow_interp_encoder_layer5(flow_interp_encoder_out4_pool)
            flow_interp_encoder_out5_pool = F.avg_pool2d(flow_interp_encoder_out5, 2, stride=2)

            flow_interp_bottleneck_out = self.flow_interp_bottleneck(flow_interp_encoder_out5_pool)
            flow_interp_bottleneck_out = torch.cat((flow_pred_bottleneck_out,
                                                    flow_interp_bottleneck_out), dim=1)
        else:
            flow_interp_encoder_out1 = self.flow_interp_encoder_layer1(in_data)
            flow_interp_encoder_out1_pool = F.avg_pool2d(flow_interp_encoder_out1, 2, stride=2)

            flow_interp_encoder_out2 = self.flow_interp_encoder_layer2(flow_interp_encoder_out1_pool)
            flow_interp_encoder_out3 = self.flow_interp_encoder_layer3(flow_interp_encoder_out2)
            flow_interp_encoder_out4 = self.flow_interp_encoder_layer4(flow_interp_encoder_out3)
            flow_interp_encoder_out5 = self.flow_interp_encoder_layer5(flow_interp_encoder_out4)

            flow_interp_bottleneck_out = self.flow_interp_bottleneck(flow_interp_encoder_out5)
            flow_interp_bottleneck_out = F.avg_pool2d(flow_interp_bottleneck_out, 2, stride=2)
            flow_interp_bottleneck_out = torch.cat((flow_pred_bottleneck_out,
                                                    flow_interp_bottleneck_out), dim=1)

        flow_interp_decoder_out5 = self.flow_interp_decoder_layer5(flow_interp_bottleneck_out)
        flow_interp_decoder_out5_cat = torch.cat((flow_interp_encoder_out5, flow_interp_decoder_out5), dim=1)

        flow_interp_decoder_out4 = self.flow_interp_decoder_layer4(flow_interp_decoder_out5_cat)
        flow_interp_decoder_out4_cat = torch.cat((flow_interp_encoder_out4, flow_interp_decoder_out4), dim=1)

        flow_interp_decoder_out3 = self.flow_interp_decoder_layer3(flow_interp_decoder_out4_cat)
        flow_interp_decoder_out3_cat = torch.cat((flow_interp_encoder_out3, flow_interp_decoder_out3), dim=1)

        flow_interp_decoder_out2 = self.flow_interp_decoder_layer2(flow_interp_decoder_out3_cat)
        flow_interp_decoder_out2_cat = torch.cat((flow_interp_encoder_out2, flow_interp_decoder_out2), dim=1)

        flow_interp_decoder_out1 = self.flow_interp_decoder_layer1(flow_interp_decoder_out2_cat)
        flow_interp_decoder_out1_cat = torch.cat((flow_interp_encoder_out1, flow_interp_decoder_out1), dim=1)
        
        flow_interp_motion_rep = self.flow_interp_refine_layer(flow_interp_decoder_out1_cat)
        
        if self.hc:
            flow_interp_motion_rep = torch.cat((
                flow_interp_motion_rep,
                flow_interp_decoder_out1,
                F.interpolate(flow_interp_decoder_out2, scale_factor= 2, mode='bilinear',align_corners=False),
                F.interpolate(flow_interp_decoder_out3, scale_factor= 4, mode='bilinear',align_corners=False),
                F.interpolate(flow_interp_decoder_out4, scale_factor= 8, mode='bilinear',align_corners=False),
                # F.interpolate(flow_interp_decoder_out5, scale_factor= 16, mode='bilinear',align_corners=False),
            ), 1)

        flow_interp_forward_flow = self.flow_interp_forward_out_layer(flow_interp_motion_rep)
        flow_interp_backward_flow = self.flow_interp_backward_out_layer(flow_interp_motion_rep)

        flow_interp_vis_map = self.flow_interp_vis_layer(flow_interp_motion_rep)
        flow_interp_vis_map = torch.sigmoid(flow_interp_vis_map)

        return flow_interp_forward_flow, flow_interp_backward_flow, flow_interp_vis_map

    def make_flow_prediction(self, x):
        if self.backbone is None:
            encoder_out1 = self.flow_pred_encoder_layer1(x)
            encoder_out1_pool = F.avg_pool2d(encoder_out1, 2, stride=2)

            encoder_out2 = self.flow_pred_encoder_layer2(encoder_out1_pool)
            encoder_out2_pool = F.avg_pool2d(encoder_out2, 2, stride=2)

            encoder_out3 = self.flow_pred_encoder_layer3(encoder_out2_pool)
            encoder_out3_pool = F.avg_pool2d(encoder_out3, 2, stride=2)

            encoder_out4 = self.flow_pred_encoder_layer4(encoder_out3_pool)
            encoder_out4_pool = F.avg_pool2d(encoder_out4, 2, stride=2)

            encoder_out5 = self.flow_pred_encoder_layer5(encoder_out4_pool)
            encoder_out5_pool = F.avg_pool2d(encoder_out5, 2, stride=2)

            bottleneck_out = self.flow_pred_bottleneck(encoder_out5_pool)
        else:
            encoder_out1 = self.flow_pred_encoder_layer1(x)
            encoder_out1_pool = F.avg_pool2d(encoder_out1, 2, stride=2)

            encoder_out2 = self.flow_pred_encoder_layer2(encoder_out1_pool)
            encoder_out3 = self.flow_pred_encoder_layer3(encoder_out2)
            encoder_out4 = self.flow_pred_encoder_layer4(encoder_out3)
            encoder_out5 = self.flow_pred_encoder_layer5(encoder_out4)

            bottleneck_out = self.flow_pred_bottleneck(encoder_out5)
            bottleneck_out = F.avg_pool2d(bottleneck_out, 2, stride=2)

        decoder_out5 = self.flow_pred_decoder_layer5(bottleneck_out)
        decoder_out5_cat = torch.cat((encoder_out5, decoder_out5), dim=1)

        decoder_out4 = self.flow_pred_decoder_layer4(decoder_out5_cat)
        decoder_out4_cat = torch.cat((encoder_out4, decoder_out4), dim=1)

        decoder_out3 = self.flow_pred_decoder_layer3(decoder_out4_cat)
        decoder_out3_cat = torch.cat((encoder_out3, decoder_out3), dim=1)

        decoder_out2 = self.flow_pred_decoder_layer2(decoder_out3_cat)
        decoder_out2_cat = torch.cat((encoder_out2, decoder_out2), dim=1)

        decoder_out1 = self.flow_pred_decoder_layer1(decoder_out2_cat)
        decoder_out1_cat = torch.cat((encoder_out1, decoder_out1), dim=1)

        motion_rep = self.flow_pred_refine_layer(decoder_out1_cat)
        
        if self.hc:
            motion_rep = torch.cat((
                motion_rep,
                decoder_out1,
                F.interpolate(decoder_out2, scale_factor= 2, mode='bilinear',align_corners=False),
                F.interpolate(decoder_out3, scale_factor= 4, mode='bilinear',align_corners=False),
                F.interpolate(decoder_out4, scale_factor= 8, mode='bilinear',align_corners=False),
                # F.interpolate(decoder_out5, scale_factor= 16, mode='bilinear',align_corners=False),
            ), 1)

        uvf = self.forward_flow_conv(motion_rep)
        uvb = self.backward_flow_conv(motion_rep)

        return uvf, bottleneck_out, uvb

    def forward(self, inputs, target_index):
        if 'image' in inputs:
            inputs = inputs['image']

        if self.training:
            self.resample2d = self.resample2d_train
        else:
            _, _, height, width = inputs[0].shape
            self.resample2d = MyResample2D(width, height).cuda()
            
        # Normalize inputs
        im1, im_target, im2 = [(im - self.mean_pix) for im in inputs]

        # Estimate bi-directional optical flows between input low FPS frame pairs
        # Downsample images for robust intermediate flow estimation
        ds_im1 = F.interpolate(im1, scale_factor=1./self.scale, mode='bilinear', align_corners=False)
        ds_im2 = F.interpolate(im2, scale_factor=1./self.scale, mode='bilinear', align_corners=False)

        uvf, bottleneck_out, uvb = self.make_flow_prediction(torch.cat((ds_im1, ds_im2), dim=1))

        uvf = self.scale * F.interpolate(uvf, scale_factor=self.scale, mode='bilinear', align_corners=False)
        uvb = self.scale * F.interpolate(uvb, scale_factor=self.scale, mode='bilinear', align_corners=False)
        bottleneck_out = F.interpolate(bottleneck_out, scale_factor=self.scale, mode='bilinear', align_corners=False)

        t = self.tlinespace[target_index]
        t = t.reshape(t.shape[0], 1, 1, 1)

        uvb_t_raw = - (1 - t) * t * uvf + t * t * uvb
        uvf_t_raw = (1 - t) * (1 - t) * uvf - (1 - t) * t * uvb

        im1w_raw = self.resample2d(im1, uvb_t_raw)  # im1w_raw
        im2w_raw = self.resample2d(im2, uvf_t_raw)  # im2w_raw

        # Perform intermediate bi-directional flow refinement
        uv_t_data = torch.cat((im1, im2, im1w_raw, uvb_t_raw, im2w_raw, uvf_t_raw), dim=1)
        uvf_t, uvb_t, t_vis_map = self.make_flow_interpolation(uv_t_data, bottleneck_out)

        uvb_t = uvb_t_raw + uvb_t # uvb_t
        uvf_t = uvf_t_raw + uvf_t # uvf_t

        im1w = self.resample2d(im1, uvb_t)  # im1w
        im2w = self.resample2d(im2, uvf_t)  # im2w

        # Compute final intermediate frame via weighted blending
        alpha1 = (1 - t) * t_vis_map
        alpha2 = t * (1 - t_vis_map)
        denorm = alpha1 + alpha2 + 1e-10
        im_t_out = (alpha1 * im1w + alpha2 * im2w) / denorm

        # Calculate training loss
        losses = {}
        losses['pix_loss'] = self.L1_loss(im_t_out, im_target)

        im_t_out_features = self.vgg16_features(im_t_out/255.)
        im_target_features = self.vgg16_features(im_target/255.)
        losses['vgg16_loss'] = self.L2_loss(im_t_out_features, im_target_features)

        losses['warp_loss'] = self.L1_loss(im1w_raw, im_target) + self.L1_loss(im2w_raw, im_target) + \
            self.L1_loss(self.resample2d(im1, uvb.contiguous()), im2) + \
            self.L1_loss(self.resample2d(im2, uvf.contiguous()), im1)

        smooth_bwd = self.L1_loss(uvb[:, :, :, :-1], uvb[:, :, :, 1:]) + \
            self.L1_loss(uvb[:, :, :-1, :], uvb[:, :, 1:, :])
        smooth_fwd = self.L1_loss(uvf[:, :, :, :-1], uvf[:, :, :, 1:]) + \
            self.L1_loss(uvf[:, :, :-1, :], uvf[:, :, 1:, :])

        losses['smooth_loss'] = smooth_bwd + smooth_fwd

        # Coefficients for total loss determined empirically using a validation set
        losses['tot'] = self.pix_alpha * losses['pix_loss'] + self.warp_alpha * losses['warp_loss'] \
            + self.vgg16_alpha * losses['vgg16_loss'] + self.smooth_alpha * losses['smooth_loss']

        # Converts back to (0, 255) range
        im_t_out = im_t_out + self.mean_pix
        im_target = im_target + self.mean_pix

        return losses, im_t_out, im_target
