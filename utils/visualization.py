from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from datasets import CLASS_COLORS, IGNORE_INDEX
from datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    restored = (image.cpu() * std + mean).clamp(0, 1)
    return restored.permute(1, 2, 0).numpy()


def colorize_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    array = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    colored = np.full((*array.shape, 3), [128, 128, 128], dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        colored[array == class_id] = color
    colored[array == IGNORE_INDEX] = [128, 128, 128]
    return colored


def save_prediction_panel(
    image: torch.Tensor,
    target: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [("Image", denormalize_image(image)), ("Ground Truth", colorize_mask(target))]
    panels.extend((name, colorize_mask(mask)) for name, mask in predictions.items())
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(history: list[dict], output_path: str | Path) -> None:
    if not history:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_miou"] * 100 for row in history], label="mIoU")
    axes[1].plot(epochs, [row["val_pixel_accuracy"] * 100 for row in history], label="Pixel Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
