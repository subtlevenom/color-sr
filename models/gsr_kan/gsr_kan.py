import torch
import torch.nn as nn
import torch.nn.functional as F


class GSRKan(nn.Module):

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super(GSRKan, self).__init__()

        self.encoder = encoder
        self.head = head

    def create_mesh(self, x:torch.Tensor, scale:int) -> torch.Tensor:
        B,C,H,W = x.shape
        xs = torch.linspace(0, 1, steps=int(scale))
        ys = torch.linspace(0, 1, steps=int(scale))
        y = torch.meshgrid(xs, xs, indexing='xy')
        y = torch.stack(y)
        y = y.repeat(B,1,H,W)
        return y

    def forward_encoder(self, x) -> torch.Tensor:
        return self.encoder(x)

    def forward_head(self, x, w) -> torch.Tensor:
        return self.head(x, w)

    def forward(self, x: torch.Tensor, scale:int=2):

        upsample = torch.nn.Upsample(scale_factor=int(scale))
        
        w = self.forward_encoder(x)
        w = w.repeat_interleave(repeats=int(scale),dim=-1)
        w = w.repeat_interleave(repeats=int(scale),dim=-2)
        
        m = self.create_mesh(x,scale).to(w.device)
        s = upsample(x)
        y = s + 2 * self.forward_head(m, w) - 1

        return y
