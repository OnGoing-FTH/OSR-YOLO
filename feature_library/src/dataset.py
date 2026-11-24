from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class ImageDataset(Dataset):
    """Deterministic RGB resize dataset that keeps source paths for indexing."""

    def __init__(self, paths: list[Path], input_size: int):
        self.paths = paths
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            original = image.copy()
            resized = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(resized)
        return {
            "image": tensor,
            "path": str(path),
            "original": TF.to_tensor(original),
        }


def collate_images(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "path": [item["path"] for item in batch],
        "original": [item["original"] for item in batch],
    }
