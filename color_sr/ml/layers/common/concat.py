from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..feed import FFN


class Concat(torch.nn.Module):

    def forward(self, *x: List[torch.Tensor], dim:int=1) -> torch.Tensor:
        x = torch.cat(x, dim=dim)
        return x
