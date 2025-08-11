from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from ..kan.sep_kan_layer import SepKANLayer2D


class MultiKANHead(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 hidden_channels: List[int] = [3,3,3],
                 grid_size: int = 5,
                 spline_order: int = 3,
                 residual_std: float = 0.1,
                 grid_range: list = [0, 1]):
        super(MultiKANHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.decoders = nn.ModuleList([
            SepKANLayer2D(
                in_channels=in_channels,
                out_channels=out_ch,
                grid_size=grid_size,
                spline_order=spline_order,
                residual_std=residual_std,
                grid_range=grid_range,
            ) for out_ch in hidden_channels
        ])

        self.kan_channels = [kan.size for kan in self.decoders]

        proj_in_channels = sum(hidden_channels)
        self.proj = nn.Conv2d(in_channels = proj_in_channels,
                              out_channels = out_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, w: List[torch.Tensor]):

        y = []
        for kan, weights in zip(self.decoders, w):
            y.append(kan(x, weights))
        y = torch.cat(y, dim=1)
        y = self.proj(y)
        return y
