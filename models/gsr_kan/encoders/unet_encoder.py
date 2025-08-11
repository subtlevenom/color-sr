from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import math
from torchvision.transforms import Normalize
import segmentation_models_pytorch as smp


class UnetEncoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 backbone='efficientnet-b2'):
        super(UnetEncoder, self).__init__()

        self.unet = smp.Unet(
            encoder_name=backbone,  # 'efficientnet-b4', 'resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            activation='sigmoid',
            in_channels=in_channels,
        )

        proj_channels = [3,32,24] #[3,32,24,48,120,352]
        self.projs = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=hid_ch,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for in_ch, hid_ch in zip(proj_channels, hidden_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        x = self.unet.encoder(x)
        y = [proj(f) for proj, f in zip(self.projs, x[:len(self.projs)])]
        y = [F.interpolate(x, (H, W), mode='bicubic') for x in y]
        return {'w': y, 'v': None}
