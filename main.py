import argparse
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
from color_sr.core import logger
from color_sr import cli


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        'Color Transfer Translation', 
        formatter_class=cli.RichHelpFormatter
    )
    subparser = parser.add_subparsers(title='Tools', required=True)

    cli.register_parsers(subparser)

    return parser.parse_args()


@hydra.main(version_base="1.3", config_path=".")
def main(cfg: DictConfig) -> Optional[float]:
    modules = cli.register_parsers(cfg)
    modules[0].main(cfg)
    args = parse_arguments()
    args.func(args)


if __name__ == '__main__':
    main()
