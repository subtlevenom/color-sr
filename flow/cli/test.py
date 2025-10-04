import argparse
from omegaconf import DictConfig
import yaml
from ..core.selector import (
    ModelSelector,
    DataSelector,
    PipelineSelector
)
from ..core.config import Config
import lightning as L
import os
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from color_sr.ml.callbacks import GenerateCallback
from lightning.pytorch.loggers import CSVLogger
from color_sr import cli
from .utils import print_rich


def main(config: DictConfig) -> None:
    print_rich(config)

    dm = DataSelector.select(config)
    model = ModelSelector.select(config)
    pipeline = PipelineSelector.select(config, model)

    logger = CSVLogger(
        save_dir=os.path.join(config.save_dir, config.experiment),
        name='logs',
        version='',
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=os.path.join(config.save_dir, config.experiment),
        max_epochs=config.pipeline.params.epochs,
        accelerator=config.accelerator,
        devices='auto',
        callbacks=[
            RichProgressBar(),
            GenerateCallback(
                every_n_epochs=1,
            ),
        ],
    )

    ckpt_path = os.path.join(config.save_dir, config.experiment, 'logs/checkpoints/last.ckpt')

    if not os.path.exists(ckpt_path):
        ckpt_path = None

    trainer.test(
        model=pipeline, 
        datamodule=dm,
        ckpt_path=ckpt_path,
    )
