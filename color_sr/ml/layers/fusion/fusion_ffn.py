from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class IlluminationEstimator(nn.Module):

    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(IlluminationEstimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in,
                               n_fea_middle,
                               kernel_size=1,
                               bias=True)

        self.depth_conv = nn.Conv2d(n_fea_middle,
                                    n_fea_middle,
                                    kernel_size=5,
                                    padding=2,
                                    bias=True,
                                    groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle,
                               n_fea_out,
                               kernel_size=1,
                               bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img, mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """

    def __init__(self, in_channels, hidden_channels=None, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.pointwise1 = nn.Conv2d(in_channels,
                                    hidden_channels,
                                    kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_channels,
                                   hidden_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   dilation=1,
                                   groups=hidden_channels)
        self.pointwise2 = nn.Conv2d(hidden_channels,
                                    out_channels,
                                    kernel_size=1)
        self.act_layer = nn.Sigmoid()

    def forward(self, x):
        y = self.pointwise1(x)
        y = self.depthwise(y)
        y = self.act_layer(y)
        x = self.pointwise2(x*y)
        return x


class FusionFFN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        **kwargs: Any,
    ):
        super(FusionFFN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        hidden_channels = sum(in_channels)

        # self.norms = nn.ModuleList([LayerNorm(ch) for ch in in_channels])

        self.ffn = FFN(in_channels=hidden_channels, out_channels=out_channels)

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:

        # x = [norm(y) for norm, y in zip(self.norms, x)]
        x = torch.cat(x, dim=1)
        x = self.ffn(x)

        return x
