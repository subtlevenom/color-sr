from pydantic import BaseModel
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    gsr_kan = 'gsr_kan'
    hsr_kan = 'hsr_kan'


class GSREncoderType(str, Enum):
    conv = 'conv'
    unet = 'unet'
    unet_tr = 'unet_tr'


class GSRKanModelParams(BaseModel):
    encoder: GSREncoderType
    in_channels: int
    out_channels: int
    grid_size: int
    spline_order: int
    residual_std: float
    grid_range: list
    normalize: bool = True


class Model(BaseModel):
    type: ModelType
    params: Union[GSRKanModelParams]
