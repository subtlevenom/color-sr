import sys
from torch.nn import *
from openeye.ml.layers.common import Flip, Pool, concat
from openeye.ml.layers.fusion import FusionTR, FusionFFN, FusionConv
from openeye.ml.layers.feed import FFN, LinearPool
from openeye.ml.layers.encoders import DETREncoder, TimmEncoder, UnetEncoder
from .gazenet.heads import DETRHead, TRHead, KANTRHead, KANFFNHead, GazeHead
from .flow import Flow


def create_layer(name, params):
    params = params or {}
    pkg = sys.modules[__name__]
    cls = getattr(pkg, name)
    if params is None:
        return cls()
    return cls(**params)


def create_model(name: str, params: dict) -> Flow:
    names = [m.module for m in params.metadata]
    modules = {
        n: create_layer(m.model, m.get('params', {}))
        for n, m in params.modules.items() if n in names
    }
    return Flow(modules, params.metadata)
