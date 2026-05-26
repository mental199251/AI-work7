import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import SegmentationMetrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
) -> dict:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    start_time = time.perf_counter()

    progress = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": loss_sum / max(sample_count, 1),
        "seconds": time.perf_counter() - start_time,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    ignore_index: int,
) -> dict:
    model.eval()
    metrics = SegmentationMetrics(num_classes, class_names, ignore_index)
    loss_sum = 0.0
    sample_count = 0
    start_time = time.perf_counter()

    for images, targets in tqdm(loader, desc="Evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        predictions = logits.argmax(dim=1)
        metrics.update(predictions, targets)

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    elapsed_seconds = time.perf_counter() - start_time
    result = metrics.compute()
    result.update(
        {
            "loss": loss_sum / max(sample_count, 1),
            "seconds": elapsed_seconds,
            "images_per_second": sample_count / elapsed_seconds if elapsed_seconds else 0.0,
            "samples": sample_count,
        }
    )
    return result
