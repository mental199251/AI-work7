from typing import Iterable

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, class_names: Iterable[str], ignore_index: int) -> None:
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().to("cpu").view(-1).long()
        target = target.detach().to("cpu").view(-1).long()
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        encoded = self.num_classes * target[valid] + prediction[valid]
        self.confusion_matrix += torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        matrix = self.confusion_matrix.float()
        intersection = torch.diag(matrix)
        target_pixels = matrix.sum(dim=1)
        prediction_pixels = matrix.sum(dim=0)
        union = target_pixels + prediction_pixels - intersection
        iou = torch.where(union > 0, intersection / union, torch.nan)

        total = matrix.sum()
        pixel_accuracy = (intersection.sum() / total).item() if total > 0 else 0.0
        valid_iou = iou[~torch.isnan(iou)]
        miou = valid_iou.mean().item() if valid_iou.numel() else 0.0
        class_iou = {
            name: (None if torch.isnan(value) else value.item())
            for name, value in zip(self.class_names, iou)
        }
        return {
            "pixel_accuracy": pixel_accuracy,
            "miou": miou,
            "class_iou": class_iou,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
