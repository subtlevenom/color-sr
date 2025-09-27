from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
from torchvision.transforms import Normalize
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp


class UnetEncoder(nn.Module):

    def __init__(
        self,
        backbone: str = 'efficientnet-b2',
        in_channels: int = 3,
        feature_channels: int = 489,
        out_channels: int = 3,
        layers: int = 5,
        features_only: bool = False,
        **kwargs,
    ):
        super(UnetEncoder, self).__init__()

        self.in_channels = in_channels
        self.feature_channels = feature_channels
        self.out_channels = out_channels
        self.features_only = features_only

        self.unet = smp.Unet(
            encoder_name=backbone,  # 'efficientnet-b4', 'resnet18',
            encoder_depth=layers,
            encoder_weights='imagenet',
            activation='sigmoid',
            in_channels=in_channels,
            classes=out_channels,
        )

        hidden_channels = self.unet.encoder.out_channels[-1]

        if feature_channels == hidden_channels:
            self.out_proj = nn.Sequential()
        else:
            self.out_proj = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=feature_channels,
                kernel_size=1,
            )

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.unet.encoder(x)
        return x

    def decode(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = self.unet.decoder(x)
        x = self.unet.segmentation_head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        f = self.encode(x)
        x = self.out_proj(f[-1])

        if self.features_only:
            return x

        y = self.decode(f)
        return x, y
