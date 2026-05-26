import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


BACKBONE_SPECS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights, (128, 256, 512)),
    "resnet34": (models.resnet34, models.ResNet34_Weights, (128, 256, 512)),
    "resnet50": (models.resnet50, models.ResNet50_Weights, (512, 1024, 2048)),
}
SUPPORTED_UPSAMPLING = {"bilinear", "deconv"}


def _bilinear_kernel(kernel_size: int) -> torch.Tensor:
    factor = (kernel_size + 1) // 2
    center = factor - 1 if kernel_size % 2 == 1 else factor - 0.5
    coordinates = torch.arange(kernel_size, dtype=torch.float32)
    filter_1d = 1 - torch.abs(coordinates - center) / factor
    return filter_1d[:, None] * filter_1d[None, :]


class FCNResNet(nn.Module):
    """FCN-8s style decoder over a configurable ResNet encoder."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        head_init_seed: int | None = None,
        upsampling: str = "bilinear",
    ) -> None:
        super().__init__()
        if backbone not in BACKBONE_SPECS:
            raise ValueError(f"Unsupported backbone '{backbone}'. Expected one of {sorted(BACKBONE_SPECS)}.")
        if upsampling not in SUPPORTED_UPSAMPLING:
            raise ValueError(f"Unsupported upsampling '{upsampling}'. Expected one of {sorted(SUPPORTED_UPSAMPLING)}.")

        constructor, weights_class, feature_channels = BACKBONE_SPECS[backbone]
        weights = weights_class.DEFAULT if pretrained else None
        encoder = constructor(weights=weights)

        self.backbone_name = backbone
        self.upsampling = upsampling
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        layer2_channels, layer3_channels, layer4_channels = feature_channels
        self.score_layer2 = nn.Conv2d(layer2_channels, num_classes, kernel_size=1)
        self.score_layer3 = nn.Conv2d(layer3_channels, num_classes, kernel_size=1)
        self.score_layer4 = nn.Conv2d(layer4_channels, num_classes, kernel_size=1)
        if upsampling == "deconv":
            self.upscore4 = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=4, stride=2, padding=1, groups=num_classes, bias=False
            )
            self.upscore3 = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=4, stride=2, padding=1, groups=num_classes, bias=False
            )
            self.upscore2 = nn.ConvTranspose2d(
                num_classes, num_classes, kernel_size=16, stride=8, padding=4, groups=num_classes, bias=False
            )
        if head_init_seed is None:
            self._initialize_decoder()
        else:
            with torch.random.fork_rng():
                torch.manual_seed(head_init_seed)
                self._initialize_decoder()

    def _initialize_decoder(self) -> None:
        for layer in [self.score_layer2, self.score_layer3, self.score_layer4]:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if self.upsampling == "deconv":
            with torch.no_grad():
                for layer in [self.upscore4, self.upscore3, self.upscore2]:
                    bilinear_weights = _bilinear_kernel(layer.kernel_size[0]).view(1, 1, *layer.kernel_size)
                    layer.weight.copy_(bilinear_weights.repeat(layer.out_channels, 1, 1, 1))

    def _upsample(self, scores: torch.Tensor, size: torch.Size, layer: nn.Module | None = None) -> torch.Tensor:
        if self.upsampling == "bilinear":
            return F.interpolate(scores, size=size, mode="bilinear", align_corners=False)
        scores = layer(scores)
        if scores.shape[-2:] != size:
            scores = F.interpolate(scores, size=size, mode="bilinear", align_corners=False)
        return scores

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        input_size = images.shape[-2:]
        features = self.stem(images)
        features = self.layer1(features)
        layer2 = self.layer2(features)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        scores = self.score_layer4(layer4)
        scores = self._upsample(scores, layer3.shape[-2:], getattr(self, "upscore4", None))
        scores = scores + self.score_layer3(layer3)
        scores = self._upsample(scores, layer2.shape[-2:], getattr(self, "upscore3", None))
        scores = scores + self.score_layer2(layer2)
        return self._upsample(scores, input_size, getattr(self, "upscore2", None))


def build_model(model_config: dict, pretrained: bool | None = None) -> FCNResNet:
    use_pretrained = model_config["pretrained"] if pretrained is None else pretrained
    return FCNResNet(
        backbone=model_config["backbone"],
        num_classes=model_config["num_classes"],
        pretrained=use_pretrained,
        head_init_seed=model_config.get("head_init_seed"),
        upsampling=model_config.get("upsampling", "bilinear"),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_size_megabytes(model: nn.Module) -> float:
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return (parameter_bytes + buffer_bytes) / (1024**2)
