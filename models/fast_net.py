import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Stem(nn.Module):
    def __init__(self, in_channel, stem_channel):
        super(Stem, self).__init__()
        self.conv = nn.Conv3d(in_channel, stem_channel, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.norm = nn.BatchNorm3d(stem_channel)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class ResStage(nn.Module):
    def __init__(self, depth, dim_in, dim_out, stride=1):
        super(ResStage, self).__init__()
        self.res_blocks = nn.ModuleList([])
        for ind in range(depth):
            if ind == 0:
                inc = dim_in
                spatial_stride = stride
            else:
                inc = dim_out
                spatial_stride = 1
            self.res_blocks.append(ResBlock(inc, dim_out, spatial_stride))
    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spatial_stride):
        super(ResBlock, self).__init__()
        self.use_shortcut = (dim_in != dim_out) or (spatial_stride != 1)
        if self.use_shortcut:
            self.branch1_conv = nn.Conv3d(dim_in, dim_out, kernel_size=(1,1,1), stride=(1,spatial_stride,spatial_stride), bias=False)
            self.branch1_norm = nn.BatchNorm3d(num_features=dim_out)
        self.branch2 = BottleneckBlock(dim_in, dim_out, spatial_stride)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        shortcut = x
        if self.use_shortcut:
            shortcut = self.branch1_conv(x)
            shortcut = self.branch1_norm(shortcut)
        x = shortcut + self.branch2(x)
        x = self.activation(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spatial_stride):
        super(BottleneckBlock, self).__init__()
        dim_inner = dim_out // 4
        self.conv_a = nn.Conv3d(in_channels=dim_in, out_channels=dim_inner, kernel_size=(3,1,1), \
                                stride=(1,1,1), padding=(1,0,0), bias=False)
        self.conv_b = nn.Conv3d(in_channels=dim_inner, out_channels=dim_inner, kernel_size=(1,3,3), \
                                stride=(1,spatial_stride,spatial_stride), padding=(0,1,1), bias=False)
        self.conv_c = nn.Conv3d(in_channels=dim_inner, out_channels=dim_out, kernel_size=(1, 1, 1), bias=False)
        self.norm_a = nn.BatchNorm3d(num_features=dim_inner)
        self.norm_b = nn.BatchNorm3d(num_features=dim_inner)
        self.norm_c = nn.BatchNorm3d(num_features=dim_out)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.activation(self.norm_a(self.conv_a(x)))
        x = self.activation(self.norm_b(self.conv_b(x)))
        x = self.norm_c(self.conv_c(x))
        return x

class FastNet(nn.Module):
    def __init__(self, stage_depth=[3, 4, 6, 3], stem_channel=8, strides=[1, 2, 2, 2]):
        super(FastNet, self).__init__()
        self.blocks = nn.ModuleList([])
        self.blocks.append(Stem(in_channel=3, stem_channel=stem_channel))
        dim_in = stem_channel
        dim_out = stem_channel * 4
        for i in range(4):
            block = ResStage(depth=stage_depth[i], dim_in=dim_in, dim_out=dim_out, stride=strides[i])
            self.blocks.append(block)
            dim_in = dim_out
            dim_out = dim_out * 2
    def forward(self, x):
        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x