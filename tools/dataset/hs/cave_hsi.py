from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from rich.progress import Progress
import cv2
import random
import numpy as np
import imageio
import albumentations as A
from omegaconf import DictConfig
from tools.utils.concurrent import concurrent
from scipy import io

THREADS = 1

SOURCE = 'source'
TARGET = 'target'


def sample(config: DictConfig) -> None:
    """
    CAVE-HSI: https://ieee-dataport.org/documents/cave-hsi
    """

    input_dir = Path(config.input)
    output_dir = Path(config.output)

    if not input_dir.is_dir():
        raise Exception(f'No such directory: {input_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy pairs (output_dir, input_path) splitted by tasks 
    splits = {}
    for name, data in config.split.items():
        output_src_dir = output_dir.joinpath(name, SOURCE)
        output_src_dir.mkdir(parents=True, exist_ok=True)
        input_src_files = list(Path(data.source).glob('*.mat'))

        output_tgt_dir = output_dir.joinpath(name, TARGET)
        output_tgt_dir.mkdir(parents=True, exist_ok=True)
        input_tgt_files = list(Path(data.target).glob('*.mat'))

        # filter out unique names
        filenames = set([f.name for f in input_src_files]) & set([f.name for f in input_tgt_files])

        src_files = [(output_src_dir, f) for f in input_src_files if f.name in filenames]
        tgt_files = [(output_tgt_dir, f) for f in input_tgt_files if f.name in filenames]

        splits[name] = src_files + tgt_files

    # Copy concurrent
    tasks = []
    with Progress() as progress:
        for split_name, split_data in splits.items():
            pb = progress.add_task(f"[cyan]{split_name}", total=len(split_data))
            with ThreadPoolExecutor(max_workers=THREADS) as executor:
                split_tasks = [
                    _copy_data(executor,
                                  output_dir=copy[0],
                                  input_file=copy[1],
                                  params=config.params)
                    for copy in split_data
                ]
                for task in split_tasks:
                    task.add_done_callback(
                        lambda _: progress.update(pb, advance=1))
                tasks.extend(split_tasks)

    _, not_done = wait(tasks, return_when=ALL_COMPLETED)

    if len(not_done) > 0:
        print(f'[Warn] Skipped {len(not_done)} image pairs.')


@concurrent
def _copy_data(
    output_dir: Path,
    input_file: Path,
    params,
):
    if not input_file.is_file():
        raise Exception('No source file')

    crop_size = params.get('crop_size', None)
    random_crop = None if crop_size is None else A.RandomCrop(height=crop_size, width=crop_size)
    n_crops = 1 if random_crop is None else params.get('n_crops', 1)

    image = io.loadmat(input_file)['hsi']
    for i in range(n_crops):
        try:
            if random_crop:
                save_name = input_file.stem + f'_{i}'
                img = random_crop(image=image)['image']
            else:
                save_name = input_file.stem
                img = image
            save_path = output_dir.joinpath(save_name)
            np.save(save_path, img)
        except Exception as e:
            print(e)
