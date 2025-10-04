import sys
from torch.nn import *
from flows.ml.layers.common import Flip, Pool, concat
from flows.ml.layers.fusion import FusionTR, FusionAtt, FusionConv
from flows.ml.layers.feed import ConvBlock, ResSENet, SENet, SepConvAtt
from flows.ml.layers.encoders import CMEncoder, DETREncoder, TimmEncoder, SmpEncoder
from flows.ml.layers.heads import DETRHead, TRHead, SimpleHead
from .flow import Flow


def create_layer(name, params):
    params = params or {}
    pkg = sys.modules[__name__]
    cls = getattr(pkg, name)
    return cls(**params)


def create_model(name: str, params: dict) -> Flow:
    names = [m.module for m in params.metadata]
    modules = {
        n: create_layer(m.model, m.get('params', None))
        for n, m in params.modules.items() if n in names
    }
    return Flow(modules, params.metadata)
