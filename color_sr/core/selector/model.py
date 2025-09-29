from omegaconf import DictConfig
from ..config.model import Model
from ..config import Config
from typing import Union
from torch import nn
from color_sr.ml.models import create_model, Flow


class ModelSelector:

    def select(config: DictConfig) -> nn.Module:
        names = [m.module for m in config.model.metadata]
        modules = {
            n: create_model(m.model, m.params)
            for n, m in config.model.modules.items()
            if n in names
        }
        return Flow(modules, config.model.metadata)
