import torch
import torch.nn as nn
from .conv_sep_kan_layer import ConvSepKanLayer


class ConvSepKan(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_channels:list, out_channels:list, kernel_sizes:list, grid_size, spline_order, residual_std, grid_range):
        super(ConvSepKan, self).__init__()

        kan_channels = [s for s in zip(in_channels, out_channels)]

        self.layers = nn.ModuleList()

        for in_ch, out_ch in kan_channels:
            layer = ConvSepKanLayer(in_channels=in_ch,
                                    out_channels=out_ch,
                                    grid_size=grid_size,
                                    spline_order=spline_order,
                                    residual_std=residual_std,
                                    grid_range=grid_range)
            self.layers.append(layer)

    def forward(self, x:torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
