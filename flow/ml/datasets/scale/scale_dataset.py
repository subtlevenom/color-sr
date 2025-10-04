import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from color_sr.ml.utils.io import read_rgb_image
from typing import List
from torchvision.transforms.v2 import Compose
from color_sr.ml.transforms.pair_trransform import PairTransform


class Image2ImageDataset(Dataset):
    def __init__(self, img_paths: List[str], scale:int, transform: Compose, p_transform: PairTransform = None) -> None:
        self.img_paths = img_paths
        self.scale = scale
        self.transform = transform
        self.p_transform = p_transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.img_paths[idx]
        tgt = read_rgb_image(path)
        if self.transform is not None:
            tgt = self.transform(tgt)

        size=(tgt.shape[-2] // int(self.scale), tgt.shape[-1] // int(self.scale))
        x = F.interpolate(tgt.unsqueeze(0), size=size, mode='bicubic').squeeze(0)

        if self.p_transform is not None:
            x, y = self.p_transform(x, y)

        return x, tgt

    def __len__(self) -> int:
        return len(self.img_paths)