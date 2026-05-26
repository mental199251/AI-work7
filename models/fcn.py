import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


SUPPORTED_BACKBONES = {"resnet18", "resnet34"}


class FCNResNet(nn.Module):
    """FCN-8s style decoder over a ResNet-18 or ResNet-34 encoder."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        head_init_seed: int | None = None,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of {sorted(SUPPORTED_BACKBONES)}.")

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            encoder = models.resnet18(weights=weights)
        else:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            encoder = models.resnet34(weights=weights)

        self.backbone_name = backbone
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.score_layer2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.score_layer3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_layer4 = nn.Conv2d(512, num_classes, kernel_size=1)
        if head_init_seed is None:
            self._initialize_classifier()
        else:
            with torch.random.fork_rng():
                torch.manual_seed(head_init_seed)
                self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        for layer in [self.score_layer2, self.score_layer3, self.score_layer4]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        input_size = images.shape[-2:]
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        scores = self.score_layer4(layer4)
        scores = F.interpolate(scores, size=layer3.shape[-2:], mode="bilinear", align_corners=False)
        scores = scores + self.score_layer3(layer3)
        scores = F.interpolate(scores, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        scores = scores + self.score_layer2(layer2)
        return F.interpolate(scores, size=input_size, mode="bilinear", align_corners=False)


def build_model(model_config: dict, pretrained: bool | None = None) -> FCNResNet:
    use_pretrained = model_config["pretrained"] if pretrained is None else pretrained
    return FCNResNet(
        backbone=model_config["backbone"],
        num_classes=model_config["num_classes"],
        pretrained=use_pretrained,
        head_init_seed=model_config.get("head_init_seed"),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_size_megabytes(model: nn.Module) -> float:
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return (parameter_bytes + buffer_bytes) / (1024**2)
