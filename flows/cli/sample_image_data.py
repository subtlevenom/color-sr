import argparse
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import os
import cv2
from pathlib import Path
import random
import numpy as np
from rich.progress import Progress
from typing import List
import imageio
import asyncio
from .utils import concurrent
from flows import cli
import albumentations as A

THREADS = 1


def add_parser(subparser: argparse) -> None:
    parser = subparser.add_parser(
        "sample-image-data",
        help="Create dataset",
        formatter_class=cli.ArgumentDefaultsRichHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="Path to input source directory",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--scale",
        type=int,
        help="Scale",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join('.data', 'x3'),
        help="Path to output directory",
        required=False,
    )
    parser.add_argument(
        "-r",
        "--seed",
        type=int,
        default=42,
        help="Seed",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--crop_size",
        type=int,
        default=1200,
        help="Crop size, set 0 to skip cropping",
        required=False,
    )
    parser.set_defaults(func=sample_dataset)


@concurrent
def _prepare_data(
    input_src_img_dir: Path,
    save_src_dir: Path,
    save_ref_dir: Path,
    name: str,
    args,
):
    source_path = input_src_img_dir.joinpath(name)
    if not source_path.is_file():
        raise Exception('No source file')

    ref_size = args.crop_size
    src_size = int(args.crop_size / args.scale)
    random_crop = A.RandomCrop(height=ref_size, width=ref_size)

    image = imageio.v3.imread(source_path)
    crop_list = [image] * 5
    for (i, image) in enumerate(crop_list):
        save_name = Path(name).stem + f'_{i}' + Path(name).suffix
        # ref
        try:
            image = random_crop(image=image)['image']
            imageio.v3.imwrite(save_ref_dir.joinpath(save_name), image)
            # src
            image = cv2.resize(image, (src_size, src_size),
                            interpolation=cv2.INTER_CUBIC)
            imageio.v3.imwrite(save_src_dir.joinpath(save_name), image)
        except Exception as e:
            print(e)


def sample_dataset(args: argparse.Namespace) -> None:
    src_img_dir = Path(args.source)
    output_dir = Path(args.output)

    if not src_img_dir.is_dir():
        raise Exception(f'No such directory: {src_img_dir}')

    save_test_src_dir = output_dir.joinpath('test', 'source')
    save_test_ref_dir = output_dir.joinpath('test', 'target')
    save_val_src_dir = output_dir.joinpath('val', 'source')
    save_val_ref_dir = output_dir.joinpath('val', 'target')
    save_train_src_dir = output_dir.joinpath('train', 'source')
    save_train_ref_dir = output_dir.joinpath('train', 'target')

    save_test_src_dir.mkdir(parents=True, exist_ok=True)
    save_test_ref_dir.mkdir(parents=True, exist_ok=True)
    save_val_src_dir.mkdir(parents=True, exist_ok=True)
    save_val_ref_dir.mkdir(parents=True, exist_ok=True)
    save_train_src_dir.mkdir(parents=True, exist_ok=True)
    save_train_ref_dir.mkdir(parents=True, exist_ok=True)

    files = list(src_img_dir.glob('*.[jpg png bmp]*'))
    random.seed(args.seed)
    random.shuffle(files)

    n = len(files)
    split = np.cumsum([int(0.7 * n), int(0.1 * n)])
    train_files = files[:split[0]]
    val_files = files[split[0]:split[1]]
    test_files = files[split[1]:]

    with Progress() as progress:
        train_pb = progress.add_task("[cyan]Train features",
                                     total=len(train_files))
        val_pb = progress.add_task("[cyan]Val images", total=len(val_files))
        test_pb = progress.add_task("[cyan]Test images", total=len(test_files))

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            train_tasks = [
                _prepare_data(
                    executor,
                    src_img_dir,
                    save_train_src_dir,
                    save_train_ref_dir,
                    filename.name,
                    args,
                ) for filename in train_files
            ]
            val_tasks = [
                _prepare_data(
                    executor,
                    src_img_dir,
                    save_val_src_dir,
                    save_val_ref_dir,
                    filename.name,
                    args,
                ) for filename in val_files
            ]
            test_tasks = [
                _prepare_data(
                    executor,
                    src_img_dir,
                    save_test_src_dir,
                    save_test_ref_dir,
                    filename.name,
                    args,
                ) for filename in test_files
            ]

            for task in train_tasks:
                task.add_done_callback(
                    lambda _: progress.update(train_pb, advance=1))
            for task in val_tasks:
                task.add_done_callback(
                    lambda _: progress.update(val_pb, advance=1))
            for task in test_tasks:
                task.add_done_callback(
                    lambda _: progress.update(test_pb, advance=1))

            _, not_done = wait(train_tasks + val_tasks + test_tasks,
                               return_when=ALL_COMPLETED)

            if len(not_done) > 0:
                print(f'[Warn] Skipped {len(not_done)} image pairs.')
