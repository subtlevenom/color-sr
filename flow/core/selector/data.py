
from omegaconf import DictConfig
from openeye.ml.datasets.kfold.datamodule import KFoldDataModule
from ..config.data import DataType
from ..config import Config
from typing import Union
from lightning import LightningDataModule
from openeye.ml.datasets import create_dataset


class DataSelector:

    def select(config: DictConfig) -> Union[LightningDataModule]:
        
        datamodule = create_dataset(config.name, config.params)    
        
        # N-folds data splitting
        if config.get('folds', 1) > 1:
            datamodule = KFoldDataModule(datamodule, config.folds)
        
        return datamodule

