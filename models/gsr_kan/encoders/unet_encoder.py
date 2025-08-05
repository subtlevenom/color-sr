from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
from torchvision.transforms import Normalize
import segmentation_models_pytorch as smp


class UnetEncoder(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, backbone='efficientnet-b2'):
        super(UnetEncoder, self).__init__()

        self.unet = smp.Unet(
            encoder_name= backbone, # 'efficientnet-b4', 'resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            activation='sigmoid',
            in_channels=in_channels,
            classes=out_channels,
        )

    @property
    def feature_channels(self):
        return 352 # len(self.effunet.encoder.out_channels)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.unet.encoder(x)
        return x

    def decode(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = self.unet.decoder(x)
        x = self.unet.segmentation_head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        y = self.decode(x)
        return y
