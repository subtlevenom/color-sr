from typing import List
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from torchvision.transforms import Normalize
import torch.utils.model_zoo as model_zoo
import segmentation_models_pytorch as smp
from ...detr import Backbone, Transformer, PositionEmbeddingSine


class DETrEncoder(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 hs_channels: int = 2,
                 mem_channels: int = 60,
                 d_model: int = 256,
                 nhead=6,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1):
        super(DETrEncoder, self).__init__()

        self.K = 4
        hidden_dim = self.K * self.K * d_model

        self.backbone = Backbone(name='resnet18', out_channels=hidden_dim)
        self.query_embed = nn.Embedding(1, d_model)

        N_steps = d_model // 2
        self.pos_embed = PositionEmbeddingSine(N_steps, normalize=True)

        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)

        self.proj_hs = nn.Conv2d(in_channels = d_model,
                              out_channels = hs_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)

        self.proj_mem = nn.Conv2d(in_channels = hidden_dim,
                              out_channels = mem_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0)

    def forward(self, x: torch.Tensor):

        features = self.backbone(x)
        # mem = self.proj_mem(features)
        # return {'w':mem, 'v':None}

        B, C, H, W = features.shape
        features = rearrange(features, 'b (c i j) h w -> (b h w) c i j', i=self.K, j=self.K)
        query_embed = self.query_embed.weight
        pos_embed = self.pos_embed(features)
        hs, mem = self.transformer(features, mask=None, query_embed=query_embed, pos_embed=pos_embed)

        hs = hs[0,:,0,:]

        mem = rearrange(mem, '(b h w) c i j -> b (c i j) h w', b=B,h=H,w=W)
        hs = rearrange(hs, '(b h w) c -> b c h w', b=B,h=H,w=W)

        hs = self.proj_hs(hs)
        mem = self.proj_mem(mem)

        return {'w':mem, 'v':F.sigmoid(hs)}
