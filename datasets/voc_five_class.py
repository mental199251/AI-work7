from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation


IGNORE_INDEX = 255
NUM_CLASSES = 5
CLASS_NAMES = ["background", "person", "car", "cat", "dog"]
CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [220, 20, 60],
        [30, 144, 255],
        [34, 139, 34],
        [255, 215, 0],
    ],
    dtype=np.uint8,
)

VOC_TO_TRAIN_ID = {
    0: 0,
    15: 1,
    7: 2,
    8: 3,
    12: 4,
}
SELECTED_FOREGROUND_IDS = np.array([15, 7, 8, 12], dtype=np.uint8)


def remap_voc_mask(mask: Image.Image) -> Image.Image:
    raw = np.asarray(mask, dtype=np.uint8)
    remapped = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    for voc_id, train_id in VOC_TO_TRAIN_ID.items():
        remapped[raw == voc_id] = train_id
    return Image.fromarray(remapped)


class VOCFiveClassSegmentation(Dataset):
    """PASCAL VOC segmentation subset for background, person, car, cat and dog."""

    def __init__(
        self,
        root: str | Path,
        image_set: str,
        transform: Optional[Callable] = None,
        year: str = "2012",
        download: bool = False,
        filter_foreground: bool = True,
    ) -> None:
        self.base_dataset = VOCSegmentation(
            root=str(root),
            year=year,
            image_set=image_set,
            download=download,
        )
        self.transform = transform
        self.filter_foreground = filter_foreground
        self.indices = self._build_indices() if filter_foreground else list(range(len(self.base_dataset)))

    def _build_indices(self) -> list[int]:
        indices: list[int] = []
        for index in range(len(self.base_dataset)):
            _, mask = self.base_dataset[index]
            raw_mask = np.asarray(mask, dtype=np.uint8)
            if np.isin(raw_mask, SELECTED_FOREGROUND_IDS).any():
                indices.append(index)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, raw_mask = self.base_dataset[self.indices[index]]
        mask = remap_voc_mask(raw_mask)
        if self.transform is None:
            raise RuntimeError("A paired image/mask transform is required for tensor output.")
        return self.transform(image, mask)
