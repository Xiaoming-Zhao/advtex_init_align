from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import cv2
import pickle
import random
import time
import argparse
import collections
import numpy as np
from PIL import Image

import torch
import torch.nn as nn


EPS = 1e-12


class TexG(nn.Module):
    def __init__(self, *, mtl_f, from_scratch=False):

        super(TexG, self).__init__()

        # mtl_imgs = cv2.imread(mtl_f)
        mtl_imgs = np.array(Image.open(mtl_f))
        assert (
            np.min(mtl_imgs) >= 0.0 and np.max(mtl_imgs) <= 255.0
        ), f"{np.min(mtl_imgs)}, {np.max(mtl_imgs)}"
        print("\nmtl_imgs: ", mtl_imgs.shape)

        assert mtl_imgs.ndim == 3, f"{mtl_imgs.shape}"

        non_black_mask = np.sum(mtl_imgs, axis=2) > 0
        # [0, 1] -> [-1, 1]
        mtl_imgs = mtl_imgs / 255.0 * 2.0 - 1.0
        # for j in range(3):
        #     mtl_imgs[:, :, j] *= non_black_mask
        mtl_imgs = np.reshape(mtl_imgs, (1, mtl_imgs.shape[0], mtl_imgs.shape[1], 3))

        raw_tex = torch.FloatTensor(mtl_imgs)
        assert (
            torch.min(raw_tex) >= -1 and torch.max(raw_tex) <= 1
        ), f"{torch.min(raw_tex)}, {torch.max(raw_tex)}"
        print("\nraw_tex: ", raw_tex.shape, "\n")

        if from_scratch:
            init_tex = 0 * torch.ones(raw_tex.shape)
            print("\nFrom scratch\n")
        else:
            init_tex = raw_tex

        self.register_parameter(
            name="tex", param=torch.nn.Parameter(data=init_tex, requires_grad=True),
        )

    @property
    def output_range(self):
        return "pm1"  # plus-minus-1

    def forward(self, placeholder):
        # return torch.clamp(self.tex, -1.0, 1.0)
        return self.tex


class CustomLeakyReLU(nn.Module):
    def __init__(self, a):
        super(CustomLeakyReLU, self).__init__()
        self.a = a

    def forward(self, x):
        return (0.5 * (1 + self.a)) * x + (0.5 * (1 - self.a)) * torch.abs(x)


class CustomConv(nn.Module):
    def __init__(self, *, in_channels, out_channels, stride):
        super(CustomConv, self).__init__()
        # Requires version >= 1.9 to support "valid" padding
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=stride, padding="valid",
        )
        torch.nn.init.normal_(self.conv.weight.data, mean=0.0, std=0.02)

    def forward(self, x):
        # NOTE:
        # - torch.pad starts from last dimension
        # - tf.pad starts from 1st dimension, and original input has foramt NHWC
        # ori_padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        # NHWC -> NCHW
        # ori_padding2 = [[0, 0], [0, 0], [1, 1], [1, 1]]
        padding = [1, 1, 1, 1, 0, 0, 0, 0]
        padded_x = torch.nn.functional.pad(x, padding, mode="constant", value=0.0,)
        x = self.conv(padded_x)
        return x


class CustomConvMask(nn.Module):
    def __init__(self, *, in_channels, stride):
        super(CustomConvMask, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, 1, kernel_size=4, stride=stride, padding="valid",
        )
        torch.nn.init.constant_(self.conv.weight.data, 1 / 16.0)

    def forward(self, x):
        # NOTE:
        # - torch.pad starts from last dimension
        # - tf.pad starts from 1st dimension, and original input has foramt NHWC
        # ori_padding = [[0, 0], [1, 1], [1, 1], [0, 0]]
        # NHWC -> NCHW
        # ori_padding2 = [[0, 0], [0, 0], [1, 1], [1, 1]]
        padding = [1, 1, 1, 1, 0, 0, 0, 0]
        padded_x = torch.nn.functional.pad(x, padding, mode="constant", value=0.0,)
        x = self.conv(padded_x)
        return x


class TexD(nn.Module):
    def __init__(self):
        super(TexD, self).__init__()

        self.n_layers = 3
        self.ndf = 64
        self.conv_layers = nn.ModuleList()
        self.conv_layers_mask = nn.ModuleList()

        self.conv1 = CustomConv(in_channels=6, out_channels=self.ndf, stride=2)
        self.conv_mask1 = CustomConvMask(in_channels=1, stride=2)

        self.lrelu = CustomLeakyReLU(0.2)

        in_channels = self.ndf
        for i in range(self.n_layers):
            out_channels = self.ndf * min(2 ** (i + 1), 2)
            stride = 1 if i == self.n_layers - 1 else 2
            tmp_conv_layer = CustomConv(
                in_channels=in_channels, out_channels=out_channels, stride=stride
            )
            tmp_conv_mask_layer = CustomConvMask(in_channels=1, stride=stride)
            self.conv_layers.append(tmp_conv_layer)
            self.conv_layers_mask.append(tmp_conv_mask_layer)

            in_channels = out_channels

        self.final_conv = CustomConv(in_channels=in_channels, out_channels=1, stride=1)
        self.final_conv_mask = CustomConvMask(in_channels=1, stride=stride)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, mask):

        # print("\n[D] input: ", input.shape, mask.shape)

        layers = []
        layers_mask = []

        convolved = self.conv1(input)
        convolved_mask = self.conv_mask1(mask)

        rectified = self.lrelu(convolved)

        layers.append(rectified)
        layers_mask.append(convolved_mask)

        for i in range(self.n_layers):
            convolved = self.conv_layers[i](layers[-1])
            convolved_mask = self.conv_layers_mask[i](layers_mask[-1])
            rectified = self.lrelu(convolved)
            layers.append(rectified)
            layers_mask.append(convolved_mask)

        convolved = self.final_conv(rectified)
        convolved_mask = self.final_conv_mask(convolved_mask)
        output = self.sigmoid(convolved)
        layers.append(output)

        output_mask = (convolved_mask > 0.1).float()
    
        return output, output_mask