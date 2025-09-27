import sys
from color_sr.ml.layers.encoders import TimmEncoder, UnetEncoder
from color_sr.ml.layers.fusion import FusionTR, FusionFFN
from color_sr.ml.layers.common import Flip
from .matrix import Matrix


def create_model(name, params):
    pkg = sys.modules[__name__]
    cls = getattr(pkg, name)
    if params is None:
        return cls()
    return cls(**params)
