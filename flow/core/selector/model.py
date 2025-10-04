from omegaconf import DictConfig
from ..config.model import Model
from ..config import Config
from typing import Union
from torch import nn
from openeye.ml.models import create_model 


class ModelSelector:

    def select(config: DictConfig) -> nn.Module:
        """Flow only model is available, name=='flow'"""
        
        model = create_model(config.name, config.params)
        return model
