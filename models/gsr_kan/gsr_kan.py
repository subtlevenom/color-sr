import torch
import torch.nn as nn
import torch.nn.functional as F


class GSRKan(nn.Module):

    def __init__(self, encoder: nn.Module, head: nn.Module, scale:int):
        super(GSRKan, self).__init__()

        self.encoder = encoder
        self.head = head
        self.scale = scale

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

    def forward(self, x: torch.Tensor, scale:int = 0):

        if scale <= 0:
            scale = self.scale

        enc:dict = self.forward_encoder(x)
        w = enc.get('w', None)
        v = enc.get('v', None)

        m = self.create_mesh(x,scale).to(x.device)

        if v is not None:
            # v = F.interpolate(v, m.shape[-2:], mode='bicubic')
            v = v.repeat_interleave(repeats=int(scale),dim=-1)
            v = v.repeat_interleave(repeats=int(scale),dim=-2)
            # m = m * F.sigmoid(v)
        
        # w = F.interpolate(w, m.shape[-2:], mode='bicubic')
        w = w.repeat_interleave(repeats=int(scale),dim=-1)
        w = w.repeat_interleave(repeats=int(scale),dim=-2)

        s = F.interpolate(x, m.shape[-2:], mode='bicubic')
        f = self.forward_head(m, w)
        y = s + f 

        return y
