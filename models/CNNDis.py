"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from torch.nn.utils.parametrizations import spectral_norm
##################################################################################
# Discriminator
##################################################################################
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize convolution
        if 'sn' in norm:
            self.conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
            norm=norm.replace('sn','')
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

from .DiffAug import DiffAugment

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self,input_dim=3,n_layer=4,num_scales=2):
        super(MsImageDis, self).__init__()
        self.n_layer = n_layer
        self.gan_type = 'hinge'
        self.dim = 64
        self.norm = 'snin'
        self.activ = 'lrelu'
        self.num_scales = num_scales
        self.pad_type = 'reflect'
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        self.use_DiffAug=True
        self.avg_loss=False

        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 4, 1, 1)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        if self.use_DiffAug:
            policy="color,translation,cutout"
            x = DiffAugment(x, policy=policy)
        outputs = []
        for model in self.cnns:
            output=model(x)
            outputs.append(output)
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        input_real.requires_grad_()
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            elif self.gan_type == 'hinge':
                loss += F.relu(1.0+out0).mean()+F.relu(1.0-out1).mean()
            elif self.gan_type == 'ralsgan':
                loss += (torch.mean((out0 - torch. mean(out1,0) - 1) ** 2) + torch.mean((out1 - torch.mean(out0,0) + 1) ** 2))/2
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
                #loss+=self.r1_reg(out1,input_real)
        if self.avg_loss:
            return loss/self.num_scales
        return loss

    def calc_gen_loss(self, input_fake,input_real):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2)# LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'hinge':
                loss += -out0.mean()
            elif self.gan_type == 'ralsgan':
                loss += (torch.mean((out0 - torch.mean(out1,0) + 1) ** 2) + torch.mean((out1 - torch.mean(out0,0) - 1) ** 2))/2
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        if self.avg_loss:
            return loss/self.num_scales
        return loss