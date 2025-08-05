from ..config.model import ModelType
from ..config import Config
from typing import Union
from color_sr.ml.models import (GSRKanModel, UnetEncoder, KANHead)


class ModelSelector:

    def select(config: Config) -> Union[GSRKanModel]:
        match config.model.type:
            case ModelType.gsr_kan:
                head = KANHead(in_channels=2,
                               out_channels=config.model.params.out_channels,
                               grid_size=config.model.params.grid_size,
                               spline_order=config.model.params.spline_order,
                               residual_std=config.model.params.residual_std,
                               grid_range=config.model.params.grid_range)
                encoder = UnetEncoder(in_channels=config.model.params.in_channels, out_channels=head.kan_size)
                return GSRKanModel(encoder=encoder, head=head)
            case _:
                raise ValueError(f'Unupported model type f{config.model.type}')
