import random
from typing import Sequence

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    return TF.pil_to_tensor(mask).squeeze(0).long()


class SegmentationTrainTransform:
    def __init__(
        self,
        size: Sequence[int],
        scale_range: Sequence[float],
        horizontal_flip_probability: float,
        ignore_index: int,
    ) -> None:
        self.size = tuple(size)
        self.scale_range = tuple(scale_range)
        self.horizontal_flip_probability = horizontal_flip_probability
        self.ignore_index = ignore_index

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        scale = random.uniform(*self.scale_range)
        scaled_height = max(1, int(round(image.height * scale)))
        scaled_width = max(1, int(round(image.width * scale)))
        resize_size = [scaled_height, scaled_width]
        image = TF.resize(image, resize_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, resize_size, interpolation=InterpolationMode.NEAREST)

        target_height, target_width = self.size
        pad_right = max(target_width - image.width, 0)
        pad_bottom = max(target_height - image.height, 0)
        if pad_right or pad_bottom:
            image = TF.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
            mask = TF.pad(mask, [0, 0, pad_right, pad_bottom], fill=self.ignore_index)

        top = random.randint(0, image.height - target_height)
        left = random.randint(0, image.width - target_width)
        image = TF.crop(image, top, left, target_height, target_width)
        mask = TF.crop(mask, top, left, target_height, target_width)

        if random.random() < self.horizontal_flip_probability:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return image_tensor, _mask_to_tensor(mask)


class SegmentationEvalTransform:
    def __init__(self, size: Sequence[int]) -> None:
        self.size = tuple(size)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, list(self.size), interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, list(self.size), interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        return image_tensor, _mask_to_tensor(mask)


def build_train_transform(data_config: dict) -> SegmentationTrainTransform:
    augmentation = data_config["augmentation"]
    return SegmentationTrainTransform(
        size=data_config["input_size"],
        scale_range=augmentation["scale_range"],
        horizontal_flip_probability=augmentation["horizontal_flip_probability"],
        ignore_index=data_config["ignore_index"],
    )


def build_eval_transform(data_config: dict) -> SegmentationEvalTransform:
    return SegmentationEvalTransform(size=data_config["input_size"])
