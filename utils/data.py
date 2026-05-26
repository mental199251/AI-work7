import torch
from torch.utils.data import DataLoader

from datasets import VOCFiveClassSegmentation, build_eval_transform, build_train_transform


def build_dataset(config: dict, split: str, is_train: bool, download: bool = False) -> VOCFiveClassSegmentation:
    data_config = config["data"]
    transform = build_train_transform(data_config) if is_train else build_eval_transform(data_config)
    return VOCFiveClassSegmentation(
        root=data_config["root"],
        year=str(data_config["year"]),
        image_set=split,
        transform=transform,
        download=download,
        filter_foreground=data_config["filter_foreground"],
    )


def build_dataloader(config: dict, dataset: VOCFiveClassSegmentation, is_train: bool) -> DataLoader:
    data_config = config["data"]
    batch_size = config["train"]["batch_size"] if is_train else config["evaluation"]["batch_size"]
    num_workers = data_config["num_workers"]
    generator = torch.Generator()
    generator.manual_seed(config["runtime"]["seed"] + (0 if is_train else 1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=data_config["pin_memory"],
        persistent_workers=data_config["persistent_workers"] and num_workers > 0,
        drop_last=is_train,
        generator=generator,
    )
