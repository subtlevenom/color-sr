from typing import Any, Callable, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..common import Pool


class LinearPool(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        **kwargs: Any,
    ):
        super(LinearPool, self).__init__()

        mid_channels = mid_channels or in_channels

        self.pool = Pool()
        self.feed = nn.Sequential(
            nn.Linear(in_features=in_channels,
                      out_features=mid_channels),
            nn.ReLU(),
            nn.Linear(in_features=mid_channels,
                      out_features=out_channels),
        )

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        x = self.feed(x)
        return x.flatten(start_dim=1)
