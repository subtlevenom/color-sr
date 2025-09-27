import sys
from openeye.ml.layers.fusion import FusionTR, FusionFFN
from openeye.ml.layers.common import Flip
from .gazenet.encoders import DETREncoder, TimmEncoder, UnetEncoder
from .gazenet.heads import DETRHead, TRHead, KANHead, SimpleHead, GazeHead
from .meta import Meta


def create_model(name, params):
    pkg = sys.modules[__name__]
    cls = getattr(pkg, name)
    if params is None:
        return cls()
    return cls(**params)
