import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ptnet import PTNet


class GSRKan(nn.Module):

    def __init__(self, encoder: nn.Module, head: nn.Module, scale:int):
        super(GSRKan, self).__init__()
        
        self.encoder = encoder
        self.head = head
        self.scale = scale

    def forward_encoder(self, x) -> torch.Tensor:
        return self.encoder(x)

    def forward_head(self, x, w) -> torch.Tensor:
        return self.head(x, w)

    def forward(self, x: torch.Tensor, scale:int = 0):

        if scale <= 0:
            scale = self.scale

        B,C,H,W = x.shape
        H_4,W_4 = H // 2, W // 2
        H2,W2 = 2*H, 2*W
        H4,W4 = 4*H, 4*W
        x_4 = F.interpolate(x, (H_4, W_4), mode='bicubic')
        x_4_s = F.interpolate(x_4, (H, W), mode='bicubic')
        dx = x - x_4_s
        dx = F.interpolate(dx, (H4, W4), mode='bicubic')
        x4 = F.interpolate(x, (H4, W4), mode='bicubic')
        
        enc:dict = self.forward_encoder(x4)

        w = enc.get('w', None)

        y = self.head(dx,w)

        y = x4 + y

        return y 
