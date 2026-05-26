from __future__ import annotations

import torch
from torch import nn


class PreActBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(inputs))
        shortcut = self.shortcut(out if not isinstance(self.shortcut, nn.Identity) else inputs)
        out = self.conv1(out)
        out = self.conv2(self.drop(self.act2(self.bn2(out))))
        return out + shortcut


class PreActResNetTiny(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.1, widths: tuple[int, int, int] = (32, 64, 128)):
        super().__init__()
        c1, c2, c3 = widths
        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.stage1 = nn.Sequential(PreActBlock(c1, c1, dropout=dropout), PreActBlock(c1, c1, dropout=dropout))
        self.stage2 = nn.Sequential(PreActBlock(c1, c2, stride=2, dropout=dropout), PreActBlock(c2, c2, dropout=dropout))
        self.stage3 = nn.Sequential(PreActBlock(c2, c3, stride=2, dropout=dropout), PreActBlock(c3, c3, dropout=dropout))
        self.head = nn.Sequential(nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c3, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.stem(inputs)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return self.head(out)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pointwise = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(channels * 4, channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(inputs).permute(0, 2, 3, 1)
        out = self.pointwise(self.norm(out)).permute(0, 3, 1, 2)
        return inputs + out


class ConvNeXtMicro(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            ConvNeXtBlock(32, dropout),
            ConvNeXtBlock(32, dropout),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            ConvNeXtBlock(64, dropout),
            ConvNeXtBlock(64, dropout),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            ConvNeXtBlock(128, dropout),
            ConvNeXtBlock(128, dropout),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.LayerNorm(128), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class ConvStemViT(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.1, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.stem(inputs).flatten(2).transpose(1, 2)
        out = self.encoder(out).mean(dim=1)
        return self.head(self.drop(self.norm(out)))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.layers = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.SiLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_channels: int, stride: int, use_se: bool, dropout: float):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, expand_channels, 1, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(expand_channels, expand_channels, 3, stride=stride, padding=1, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.SiLU(inplace=True),
            SqueezeExcite(expand_channels) if use_se else nn.Identity(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(expand_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.block(inputs)
        return inputs + out if self.use_residual else out


class MobileNetV3_28(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            InvertedResidual(24, 24, 48, 1, False, dropout),
            InvertedResidual(24, 40, 72, 2, True, dropout),
            InvertedResidual(40, 40, 96, 1, True, dropout),
            InvertedResidual(40, 80, 160, 2, False, dropout),
            InvertedResidual(80, 96, 192, 1, True, dropout),
            nn.Conv2d(96, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


HETERO_MODEL_NAMES = {
    "preact_resnet_tiny",
    "wide_resnet_tiny",
    "convnext_micro",
    "convstem_vit",
    "mobilenetv3_28",
}


def build_heterogeneous_model(model_name: str, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.1) -> nn.Module:
    if model_name == "preact_resnet_tiny":
        return PreActResNetTiny(num_classes=num_classes, in_channels=in_channels, dropout=dropout, widths=(32, 64, 128))
    if model_name == "wide_resnet_tiny":
        return PreActResNetTiny(num_classes=num_classes, in_channels=in_channels, dropout=dropout, widths=(48, 96, 192))
    if model_name == "convnext_micro":
        return ConvNeXtMicro(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    if model_name == "convstem_vit":
        return ConvStemViT(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    if model_name == "mobilenetv3_28":
        return MobileNetV3_28(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    raise ValueError(f"unknown heterogeneous model_name={model_name!r}")
