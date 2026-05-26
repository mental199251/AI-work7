from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation


IGNORE_INDEX = 255
FIVE_CLASS_NAMES = ["background", "person", "car", "cat", "dog"]
FIVE_CLASS_COLORS = np.array(
    [
        [0, 0, 0],
        [220, 20, 60],
        [30, 144, 255],
        [34, 139, 34],
        [255, 215, 0],
    ],
    dtype=np.uint8,
)
VOC21_CLASS_NAMES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_TO_TRAIN_ID = {
    0: 0,
    15: 1,
    7: 2,
    8: 3,
    12: 4,
}
SELECTED_FOREGROUND_IDS = np.array([15, 7, 8, 12], dtype=np.uint8)


def voc_colormap(length: int = 21) -> np.ndarray:
    colormap = np.zeros((length, 3), dtype=np.uint8)
    for class_id in range(length):
        value = class_id
        for shift in range(8):
            colormap[class_id, 0] |= ((value >> 0) & 1) << (7 - shift)
            colormap[class_id, 1] |= ((value >> 1) & 1) << (7 - shift)
            colormap[class_id, 2] |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
    return colormap


VOC21_CLASS_COLORS = voc_colormap()


@dataclass(frozen=True)
class SegmentationTaskSpec:
    name: str
    class_names: list[str]
    class_colors: np.ndarray
    num_classes: int


TASK_SPECS = {
    "five_class": SegmentationTaskSpec("five_class", FIVE_CLASS_NAMES, FIVE_CLASS_COLORS, 5),
    "voc21": SegmentationTaskSpec("voc21", VOC21_CLASS_NAMES, VOC21_CLASS_COLORS, 21),
}

# Backward compatible exports for the original five-class experiment.
NUM_CLASSES = TASK_SPECS["five_class"].num_classes
CLASS_NAMES = TASK_SPECS["five_class"].class_names
CLASS_COLORS = TASK_SPECS["five_class"].class_colors


def get_task_spec(task: str) -> SegmentationTaskSpec:
    if task not in TASK_SPECS:
        raise ValueError(f"Unsupported segmentation task '{task}'. Expected one of {sorted(TASK_SPECS)}.")
    return TASK_SPECS[task]


def remap_voc_mask(mask: Image.Image, task: str = "five_class") -> Image.Image:
    raw = np.asarray(mask, dtype=np.uint8)
    if task == "voc21":
        return Image.fromarray(raw.copy())
    get_task_spec(task)
    remapped = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    for voc_id, train_id in VOC_TO_TRAIN_ID.items():
        remapped[raw == voc_id] = train_id
    return Image.fromarray(remapped)


class VOCSegmentationTask(Dataset):
    """PASCAL VOC semantic segmentation with configurable label tasks."""

    def __init__(
        self,
        root: str | Path,
        image_set: str,
        transform: Optional[Callable] = None,
        year: str = "2012",
        download: bool = False,
        filter_foreground: bool = True,
        task: str = "five_class",
    ) -> None:
        self.task_spec = get_task_spec(task)
        self.task = task
        self.class_names = self.task_spec.class_names
        self.class_colors = self.task_spec.class_colors
        self.num_classes = self.task_spec.num_classes
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
            foreground_ids = SELECTED_FOREGROUND_IDS if self.task == "five_class" else np.arange(1, 21)
            if np.isin(raw_mask, foreground_ids).any():
                indices.append(index)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, raw_mask = self.base_dataset[self.indices[index]]
        mask = remap_voc_mask(raw_mask, self.task)
        if self.transform is None:
            raise RuntimeError("A paired image/mask transform is required for tensor output.")
        return self.transform(image, mask)


class VOCFiveClassSegmentation(VOCSegmentationTask):
    """Backward compatible five-class VOC dataset wrapper."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["task"] = "five_class"
        super().__init__(*args, **kwargs)
