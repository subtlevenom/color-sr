from omegaconf import DictConfig
from tools.utils import text
from tools.dataset.image.sample_scale import sample_dataset


def main(config: DictConfig) -> None:
    text.print(config)
    sample_dataset(config)
