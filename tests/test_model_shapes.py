import pytest
import torch

from models import FCNResNet


@pytest.mark.parametrize(
    ("backbone", "upsampling", "num_classes"),
    [
        ("resnet18", "bilinear", 5),
        ("resnet34", "bilinear", 5),
        ("resnet50", "bilinear", 5),
        ("resnet34", "deconv", 5),
        ("resnet34", "bilinear", 21),
    ],
)
def test_model_output_matches_input_resolution(backbone: str, upsampling: str, num_classes: int) -> None:
    model = FCNResNet(backbone, num_classes, pretrained=False, upsampling=upsampling)
    images = torch.randn(2, 3, 64, 64)

    output = model(images)

    assert output.shape == (2, num_classes, 64, 64)
