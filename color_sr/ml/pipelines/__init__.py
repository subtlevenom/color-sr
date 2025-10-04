import sys, inspect
import lightning as L
from . import default
from . import gaze
from . import origin


def create_pipeline(name, model, params) -> L.LightningModule:
    params = params or {}
    modules = sys.modules[__name__]
    module = getattr(modules, name)
    for name, cls in inspect.getmembers(module):
        if inspect.isclass(cls) and issubclass(cls, L.LightningModule):
            return cls(model=model, **params)
    return None
