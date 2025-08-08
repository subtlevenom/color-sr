from pydantic import BaseModel
from enum import Enum
from typing import List, Union


class ModelType(str, Enum):
    gsr_kan = 'gsr_kan'
    hsr_kan = 'hsr_kan'


# HSR Model


class HSRModelParams(BaseModel):
    pass


# GSR Model


class EncoderType(str, Enum):
    detr = 'detr'
    unet = 'unet'
    cm = 'cm'


class HeadType(str, Enum):
    kan = 'kan'


class EncoderParams(BaseModel):
    type: EncoderType


class HeadParams(BaseModel):
    type: HeadType


class CMEncoderParams(EncoderParams):
    in_channels: int
    out_channels: int

class UnetEncoderParams(EncoderParams):
    in_channels: int
    out_channels: int


class DETrEncoderParams(EncoderParams):
    in_channels: int
    out_channels: int
    d_model: int
    nhead: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float = 0.1


class KanHeadParams(HeadParams):
    in_channels: int
    out_channels: int
    grid_size: int
    spline_order: int
    residual_std: float
    grid_range: list
    normalize: bool = True


class GSRModelParams(BaseModel):
    encoder: Union[DETrEncoderParams, CMEncoderParams]
    head: Union[KanHeadParams]


# Model


class Model(BaseModel):
    type: ModelType
    params: Union[GSRModelParams]
