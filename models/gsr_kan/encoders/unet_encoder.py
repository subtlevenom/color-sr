from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import math
from torchvision.transforms import Normalize
import segmentation_models_pytorch as smp
import timm


class UnetEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 backbone='efficientnet-b2'):
        super(UnetEncoder, self).__init__()

        self.unet = smp.create_model(
            arch='unet',
            encoder_name=backbone,  # 'efficientnet-b4', 'resnet18',
            encoder_depth=3,
            decoder_channels = (64, 32, 16), #(256, 128, 64, 32, 16),
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels
        )

        proj_channels = (64,64,128)
        self.proj = nn.ModuleList([
            nn.Conv2d(in_channels = in_ch,
                            out_channels = out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
            for in_ch in proj_channels

        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        y = self.unet.encoder(x)
        w = [self.proj[i](y[i+1]) for i in range(len(self.proj))]
        w = [F.interpolate(x, (H, W), mode='bicubic') for x in w]
        return {'w': w, 'v': None}
