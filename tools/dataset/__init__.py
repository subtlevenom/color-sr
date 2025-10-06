from omegaconf import DictConfig
from tools.utils import text
from .sr import sample_scale as sr_scale
from .hs import sample_image as hs_image


def sample(config: DictConfig) -> None:
    match (config.get('type', 1)):
        case 'sr-scale': 
            sr_scale.sample(config)
        case 'hs-image': 
            hs_image.sample(config)

