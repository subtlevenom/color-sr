from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import math
from torchvision.transforms import Normalize
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp
from ...detr import Backbone


class ConvEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        w_channels: int = 60,
        v_channels: int = 2,
    ):
        super(ConvEncoder, self).__init__()

        self.w_channels = w_channels
        self.v_channels = v_channels
        self.backbone = Backbone(name='resnet18',
                                 out_channels=w_channels + v_channels)

    def forward(self, x: torch.Tensor):

        y = self.backbone(x)
        w = y[:, :self.w_channels]
        v = y[:, self.w_channels:]
        return {'w': w, 'v': F.sigmoid(v)}
