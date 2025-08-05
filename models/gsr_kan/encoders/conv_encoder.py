from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
from torchvision.transforms import Normalize
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp


class ConvHead(nn.Module):

    def __init__(self, in_channels: int, out_channels: int = 2):
        super(ConvHead, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              1,
                              kernel_size=1,
                              stride=1,
                              padding=0)

        self.feed = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1),
        )

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape
        y = self.conv(x)
        x = self.feed(x * y)
        return x.view((B, -1))
