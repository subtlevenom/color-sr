import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from flows.ml.layers.kan import SepKANLayer2D


class KANHead(nn.Module):

    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 3,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 residual_std: float = 0.1,
                 grid_range: list = [0, 1]):
        super(KANHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kan_decoder = SepKANLayer2D(in_channels=in_channels,
                                         out_channels=out_channels,
                                         grid_size=grid_size,
                                         spline_order=spline_order,
                                         residual_std=residual_std,
                                         grid_range=grid_range)

        self.kan_size = self.kan_decoder.size

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, w: torch.Tensor):

        return self.kan_decoder(x, w)
