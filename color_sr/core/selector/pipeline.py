from ..config.pipeline import PipelineType
from ..config import Config
from typing import Union
from color_sr.ml.pipelines import (
    DefaultPipeline,
)
from color_sr.ml.models import (
    Matrix,
)


class PipelineSelector:
    def select(config: Config, model: Union[Matrix]) -> Union[DefaultPipeline]:
        match config.pipeline.type:
            case PipelineType.default:
                return DefaultPipeline(
                    model=model,
                    optimiser=config.pipeline.params.optimizer,
                    lr=config.pipeline.params.lr,
                    weight_decay=config.pipeline.params.weight_decay
                )
            case _:
                raise ValueError(f'Unupported pipeline type f{config.pipeline.type}')
