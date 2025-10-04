from omegaconf import DictConfig
import lightning as L
from torch import nn
from ..config.pipeline import PipelineType
from ..config import Config
from typing import Union
from openeye.ml.pipelines import create_pipeline


class PipelineSelector:
    def select(model: nn.Module, config: DictConfig) -> L.LightningModule:
        model = create_pipeline(config.name, model, config.params)
        return model
