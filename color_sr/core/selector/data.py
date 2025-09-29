from omegaconf import DictConfig
from ..config.data import DataType
from ..config import Config
from typing import Union
from color_sr.ml.datasets import (
    ImgDataModule,
    ScaleDataModule,
    KFoldDataModule,
)


class DataSelector:
    def select(config: DictConfig) -> Union[ImgDataModule]:
        match config.data.type:
            case DataType.scale:
                dm = ScaleDataModule(
                    train_path=config.data.train.source,
                    val_path=config.data.val.source,
                    test_path=config.data.test.source,
                    scale=config.data.scale,
                    train_batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case DataType.img:
                dm = ImgDataModule(
                    train_a=config.data.train.source,
                    train_b=config.data.train.target,
                    val_a=config.data.val.source,
                    val_b=config.data.val.target,
                    test_a=config.data.test.source,
                    test_b=config.data.test.target,
                    train_batch_size=config.pipeline.params.batch_size,
                    val_batch_size=config.pipeline.params.val_batch_size,
                    test_batch_size=config.pipeline.params.test_batch_size,
                )
            case _:
                raise ValueError(f'Unupported data type f{config.data.type}')

        if config.data.folds > 1:
            dm = KFoldDataModule(dm, config.data.folds)
        
        return dm