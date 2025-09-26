from enum import Enum
from pydantic import BaseModel
from typing import Union


class PipelineType(str, Enum):
    odc = 'odc'
    odc_face = 'odc_face'
    odc_gaze = 'odc_gaze'
    mpii = 'mpii'
    gaze360 = 'gaze360'
    gan = 'gan'
    se = 'se'
    se_swa = 'se_swa'


class PipelineParams(BaseModel):
    lr: float = 1e-3
    batch_size: int = 32
    val_batch_size: int = 1
    test_batch_size: int = 1
    predict_batch_size: int = 1
    epochs: int = 500
    monitor: str = 'val_loss'
    save_freq: int = 10
    visualize_freq: int = 10


class DefaultPipelineParams(PipelineParams):
    optimizer: str = 'adam'
    weight_decay: float = 0.0


class Pipeline(BaseModel):
    type: PipelineType = PipelineType.odc
    params: Union[
        DefaultPipelineParams,
    ] = DefaultPipelineParams()
