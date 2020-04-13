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

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory 
# from correlation_package.correlation import Correlation

from .model_utils import MyResample2D
from .shelfnet import get_shelf_unet

'''
class CostVolumn(nn.Module):
    def __init__(self, im_channel=3, feat_channel=16, max_displacement=4, flow_est_channel=32):
        super(CostVolumn, self).__init__()
        
        corr_channel = (2 * max_displacement + 1) ** 2
        trans_channel = feat_channel + 2 + corr_channel
        dense_channel = flow_est_channel + trans_channel
        
        self.conv = nn.Conv2d(im_channel, feat_channel, 3, padding=1)
        self.corr = Correlation(
            pad_size=max_displacement,
            kernel_size=1,
            max_displacement=max_displacement,
            stride1=1,
            stride2=1,
            corr_multiply=1
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.trans = nn.Conv2d(trans_channel, flow_est_channel, 3, padding=1)
        self.flow_est = nn.Conv2d(dense_channel, 2, 3, padding=1)
    
    def forward(self, ims, forward_flow, backward_flow):
        im1 = ims[:, :3, :, :]
        im2 = ims[:, 3:6, :, :]
        
        f1 = self.conv(im1)
        f2 = self.conv(im2)
        
        flow_f1 = self.warp(f2, forward_flow)
        flow_f2 = self.warp(f1, backward_flow)
        
        corr1 = self.corr(f1, flow_f1)
        corr1 = self.relu(corr1)
        
        corr2 = self.corr(f2, flow_f2)
        corr2 = self.relu(corr2)
        
        x1 = torch.cat((corr1, f1, backward_flow), 1)
        x2 = torch.cat((corr2, f2, forward_flow), 1)
        
        t1 = self.trans(x1)
        t2 = self.trans(x2)
        
        flow1 = self.flow_est(torch.cat((t1, x1), 1))
        flow2 = self.flow_est(torch.cat((t2, x2), 1))
        
        return torch.cat((flow1, flow2), 1)
        
    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = torch.Tensor(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask
'''

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class ResSubPixelSR(nn.Module):
    def __init__(self, r, input_channel, output_channel, shortcut_channel, kernel_size=3):
        super(ResSubPixelSR, self).__init__()
        
        self.r = r
        self.conv = nn.Conv2d(input_channel, self.r*self.r*output_channel, kernel_size, padding=kernel_size//2)
        self.shortcut_conv = nn.Conv2d(shortcut_channel, input_channel, 1, padding=0)
        self.ps = nn.PixelShuffle(self.r)
        
    def forward(self, shortcut, x):
        s = self.shortcut_conv(shortcut)
        
        x = s + x
        x = self.conv(x)
        x = self.ps(x)
        
        return x

class Refiner(nn.Module):
    def __init__(self, input_chan, output_chan, shortcut_chan, kernel_size=3):
        super(Refiner, self).__init__()

        self.shortcut_conv = nn.Conv2d(shortcut_chan, input_chan, kernel_size, padding=kernel_size//2)
        self.conv = nn.Conv2d(input_chan, output_chan, kernel_size, padding=kernel_size//2)

    def forward(self, shortcut, x):
        s = self.shortcut_conv(shortcut)

        x = s + x
        x = self.conv(x)

        return x

class HJSuperSloMoShelf(nn.Module):
    def __init__(self, args, mean_pix=[109.93, 109.167, 101.455], in_channel=6):
        super(HJSuperSloMoShelf, self).__init__()
        self.is_output_flow = False
        
        self.flow_pred = get_shelf_unet(input_channel=6)

        self.forward_flow_conv = ResSubPixelSR(4, 17, 2, 16)
        self.backward_flow_conv = ResSubPixelSR(4, 17, 2, 16)

        # self.flow_refine = CostVolumn()
        
        self.flow_interp = get_shelf_unet(input_channel=16)

        self.flow_interp_forward_out_layer = ResSubPixelSR(4, 17, 2, 16)
        self.flow_interp_backward_out_layer = ResSubPixelSR(4, 17, 2, 16)

        # self.interp_refine = CostVolumn()

        # visibility
        self.flow_interp_vis_layer = ResSubPixelSR(4, 17, 1, 16)

        self.vis_refine = Refiner(1, 1, 16, 3)

        self.resample2d_train = MyResample2D(args.crop_size[1], args.crop_size[0])

        mean_pix = torch.from_numpy(np.array(mean_pix)).float()
        mean_pix = mean_pix.view(1, 3, 1, 1)
        self.register_buffer('mean_pix', mean_pix)

        self.args = args
        self.scale = args.flow_scale
        
        # grad supervision
        self.grad_supervision = args.grad_supervision
        
        if self.grad_supervision:
            self.get_grad = Get_gradient()
            self.grad_alpha = 0.8
        else:
            self.grad_alpha = 0.

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

    def make_flow_interpolation(self, in_data):
        flow_interp_x0, flow_interp_output = self.flow_interp(in_data)

        flow_interp_forward_flow = self.flow_interp_forward_out_layer(flow_interp_x0, flow_interp_output)
        flow_interp_backward_flow = self.flow_interp_backward_out_layer(flow_interp_x0, flow_interp_output)
        
        '''
        ims = in_data[:, :6, :, :]
        flow_interp_forward_flow, flow_interp_backward_flow = self.interp_refine(
            ims,
            flow_interp_forward_flow,
            flow_interp_backward_flow
        )
        '''

        flow_interp_vis_map = self.flow_interp_vis_layer(flow_interp_x0, flow_interp_output)
        # flow_interp_vis_map = self.vis_refine(in_data, flow_interp_vis_map)
        flow_interp_vis_map = torch.sigmoid(flow_interp_vis_map)

        return flow_interp_forward_flow, flow_interp_backward_flow, flow_interp_vis_map

    def make_flow_prediction(self, x):
        flow_pred_x0, flow_pred_output = self.flow_pred(x)

        uvf = self.forward_flow_conv(flow_pred_x0, flow_pred_output)
        uvb = self.backward_flow_conv(flow_pred_x0, flow_pred_output)
        
        '''
        ims = x[:, :6, :, :]
        uvf, uvb = self.flow_refine(ims, uvf, uvb)
        '''
 
        return uvf, uvb

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

        uvf, uvb = self.make_flow_prediction(torch.cat((ds_im1, ds_im2), dim=1))

        uvf = self.scale * F.interpolate(uvf, scale_factor=self.scale, mode='bilinear', align_corners=False)
        uvb = self.scale * F.interpolate(uvb, scale_factor=self.scale, mode='bilinear', align_corners=False)

        t = self.tlinespace[target_index]
        t = t.reshape(t.shape[0], 1, 1, 1)

        uvb_t_raw = - (1 - t) * t * uvf + t * t * uvb
        uvf_t_raw = (1 - t) * (1 - t) * uvf - (1 - t) * t * uvb

        im1w_raw = self.resample2d(im1, uvb_t_raw)  # im1w_raw
        im2w_raw = self.resample2d(im2, uvf_t_raw)  # im2w_raw

        # Perform intermediate bi-directional flow refinement
        uv_t_data = torch.cat((im1, im2, im1w_raw, uvb_t_raw, im2w_raw, uvf_t_raw), dim=1)
        uvf_t, uvb_t, t_vis_map = self.make_flow_interpolation(uv_t_data)

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
        
        # grad supervision
        if self.grad_supervision:
            target_grad = self.get_grad(im_target)
            out_grad = self.get_grad(im_t_out)
            
            losses['grad_loss'] = self.L1_loss(out_grad, target_grad)

        # Coefficients for total loss determined empirically using a validation set
        losses['tot'] = self.pix_alpha * losses['pix_loss'] + self.warp_alpha * losses['warp_loss'] \
            + self.vgg16_alpha * losses['vgg16_loss'] + self.smooth_alpha * losses['smooth_loss'] \
            + self.grad_alpha * losses['grad_loss']

        # Converts back to (0, 255) range
        im_t_out = im_t_out + self.mean_pix
        im_target = im_target + self.mean_pix

        return losses, im_t_out, im_target
