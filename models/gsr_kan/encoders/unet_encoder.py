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
                 backbone='efficientnet-b0'):
        super(UnetEncoder, self).__init__()

        self.backbone = timm.create_model('efficientvit-mit', pretrained=True)
        
        self.unet = smp.create_model.Unet(
            encoder_name=backbone,  # 'efficientnet-b4', 'resnet18',
            encoder_depth=5,
            decoder_channels = (256, 128, 64, 64, 128),
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.unet(x)
        return {'w': y, 'v': None}
