import torch
from torch.utils.data import DataLoader

from datasets import VOCSegmentationTask, build_eval_transform, build_train_transform, get_task_spec


def build_dataset(config: dict, split: str, is_train: bool, download: bool = False) -> VOCSegmentationTask:
    data_config = config["data"]
    task = data_config.get("task", "five_class")
    task_spec = get_task_spec(task)
    if config["model"]["num_classes"] != task_spec.num_classes:
        raise ValueError(
            f"Model outputs {config['model']['num_classes']} classes but task '{task}' requires "
            f"{task_spec.num_classes}."
        )
    transform = build_train_transform(data_config) if is_train else build_eval_transform(data_config)
    return VOCSegmentationTask(
        root=data_config["root"],
        year=str(data_config["year"]),
        image_set=split,
        transform=transform,
        download=download,
        filter_foreground=data_config["filter_foreground"],
        task=task,
    )


def build_dataloader(config: dict, dataset: VOCSegmentationTask, is_train: bool) -> DataLoader:
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
