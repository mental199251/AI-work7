import pytest
import torch

from utils.metrics import SegmentationMetrics


def test_metrics_exclude_ignored_pixels() -> None:
    metrics = SegmentationMetrics(num_classes=2, class_names=["background", "object"], ignore_index=255)
    target = torch.tensor([[0, 0, 1, 1, 255]])
    prediction = torch.tensor([[0, 1, 1, 1, 0]])

    metrics.update(prediction, target)
    result = metrics.compute()

    assert result["pixel_accuracy"] == pytest.approx(0.75)
    assert result["class_iou"]["background"] == pytest.approx(0.5)
    assert result["class_iou"]["object"] == pytest.approx(2 / 3)
    assert result["miou"] == pytest.approx((0.5 + 2 / 3) / 2)
