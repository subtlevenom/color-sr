from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..feed import FFN


class FusionFFN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        **kwargs: Any,
    ):
        super(FusionFFN, self).__init__()

        self.ffn = FFN(in_channels=sum(in_channels), out_channels=out_channels)

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, dim=1)
        x = self.ffn(x)
        return x
