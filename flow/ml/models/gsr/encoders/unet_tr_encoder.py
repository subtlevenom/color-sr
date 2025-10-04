from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import math
from torchvision.transforms import Normalize
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp


class UnetTREncoder(nn.Module):

    # models: https://huggingface.co/docs/timm/models
    # model = timm.create_model('swsl_resnet18', pretrained=True)
    # config = resolve_data_config({}, model=model)
    # transform = create_transform(**config)

    def __init__(self, backbone='efficientnet-b2'):
        super(UnetTREncoder, self).__init__()

        self.unet = smp.Unet(
            encoder_name= backbone, # 'efficientnet-b4', 'resnet18',
            encoder_depth=5,
            encoder_weights='imagenet',
            activation='sigmoid',
            in_channels=3,
            classes=3,
        )

    @property
    def feature_channels(self):
        return 352 # len(self.effunet.encoder.out_channels)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.unet.encoder(x)
        return x

    def decode(self, x: List[torch.Tensor]) -> torch.Tensor:
        x = self.unet.decoder(*x)
        x = self.unet.segmentation_head(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        y = self.decode(x)
        return y
