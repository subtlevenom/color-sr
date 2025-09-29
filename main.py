import argparse
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
from color_sr.core import logger
from color_sr import cli


@hydra.main(version_base='1.1.0', config_path='config', config_name='config')
def main(cfg: DictConfig) -> Optional[float]:
    entry = cli.register_task(cfg.get('task', None))
    return entry(cfg)


if __name__ == '__main__':
    main()
