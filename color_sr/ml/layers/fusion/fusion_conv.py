from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..feed import FFN


class FusionConv(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size=1,
        **kwargs: Any,
    ):
        super(FusionConv, self).__init__()

        self.feed = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size),
        )

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, dim=1)
        x = self.feed(x)
        return x
