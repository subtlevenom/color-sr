import os
import random
import torch
import lightning as L
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    CenterCrop,
    RandomCrop,
)
from torch.utils.data import DataLoader
from typing import Tuple
from .scale_dataset import Image2ImageDataset
from color_sr.core import Logger
from color_sr.ml.transforms.pair_trransform import PairTransform

CROP = 256

class ScaleDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_path: str,
            val_path: str,
            test_path: str,
            scale:int,
            train_batch_size: int = 32,
            val_batch_size: int = 32,
            test_batch_size: int = 32,
            num_workers: int = min(12, os.cpu_count() - 1),
            img_exts: Tuple[str] = (".png", ".jpg"),
            seed: int = 42,
    ) -> None:
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.scale = scale

        random.seed(seed)

        paths = [
            os.path.join(train_path, fname)
            for fname in os.listdir(train_path)
            if fname.endswith(img_exts)
        ]
        self.train_paths = sorted(paths)

        paths = [
            os.path.join(val_path, fname)
            for fname in os.listdir(val_path)
            if fname.endswith(img_exts)
        ]
        self.val_paths = sorted(paths)

        paths = [
            os.path.join(test_path, fname)
            for fname in os.listdir(test_path)
            if fname.endswith(img_exts)
        ]

        self.test_paths = sorted(paths)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.train_image_p_transform = None
        self.val_image_p_transform = None
        self.test_image_p_transform = None

        self.image_train_transform = Compose([
            ToImage(),
            RandomCrop(600),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.image_val_transform = Compose([
            ToImage(),
            RandomCrop(600),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.image_test_transform = Compose([
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = Image2ImageDataset(
                self.train_paths, self.scale, self.image_train_transform, self.train_image_p_transform,
            )
            self.val_dataset = Image2ImageDataset(
                self.val_paths, self.scale, self.image_val_transform, self.val_image_p_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = Image2ImageDataset(
                self.test_paths, self.scale, self.image_test_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
