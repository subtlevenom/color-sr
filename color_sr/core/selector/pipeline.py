from omegaconf import DictConfig
from ..config.pipeline import PipelineType
from ..config import Config
from typing import Union
from color_sr.ml.pipelines import (
    DefaultPipeline,
)
from color_sr.ml.models import (
    Flow,
)


class PipelineSelector:
    def select(config: DictConfig, model: Union[Flow]) -> Union[DefaultPipeline]:
        match config.pipeline.type:
            case PipelineType.default:
                return DefaultPipeline(
                    model=model,
                    optimiser=config.pipeline.optimizer,
                    lr=config.pipeline.lr,
                    weight_decay=config.pipeline.weight_decay
                )
            case _:
                raise ValueError(f'Unupported pipeline type f{config.pipeline.type}')
