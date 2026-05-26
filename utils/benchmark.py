import time

import torch
from torch import nn
from torch.utils.data import Dataset


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_single_image_inference(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    warmup_iterations: int,
    measured_images: int,
) -> dict:
    model.eval()
    if len(dataset) == 0:
        raise RuntimeError("Cannot benchmark an empty validation dataset.")

    warmup_image, _ = dataset[0]
    warmup_image = warmup_image.unsqueeze(0).to(device)
    for _ in range(warmup_iterations):
        model(warmup_image)
    _synchronize(device)

    image_count = min(measured_images, len(dataset))
    total_seconds = 0.0
    for index in range(image_count):
        image, _ = dataset[index]
        image = image.unsqueeze(0).to(device)
        _synchronize(device)
        start_time = time.perf_counter()
        model(image)
        _synchronize(device)
        total_seconds += time.perf_counter() - start_time

    return {
        "benchmark_images": image_count,
        "warmup_iterations": warmup_iterations,
        "single_image_milliseconds": 1000.0 * total_seconds / image_count,
    }
