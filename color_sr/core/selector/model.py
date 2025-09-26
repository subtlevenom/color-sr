from ..config.model import Model
from ..config import Config
from typing import Union
from torch import nn
from openeye.ml.models import create_model, Meta


class ModelSelector:

    def select(config: Config) -> nn.Module:
        names = [m.module for m in config.model.metadata]
        modules = {
            n: create_model(m.model, m.params)
            for n, m in config.model.modules.items()
            if n in names
        }
        return Meta(modules, config.model.metadata)
