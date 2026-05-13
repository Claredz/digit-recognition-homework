import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class MediumCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


def normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower().replace("-", "_")
    aliases = {
        "small": "small_cnn",
        "smallcnn": "small_cnn",
        "small_cnn": "small_cnn",
        "medium": "medium_cnn",
        "mediumcnn": "medium_cnn",
        "medium_cnn": "medium_cnn",
    }
    if normalized not in aliases:
        raise ValueError(f"不支持的 model_name='{model_name}'。请使用 small_cnn 或 medium_cnn。")
    return aliases[normalized]


def build_model(
    config_or_name,
    num_classes: int | None = None,
    in_channels: int | None = None,
    dropout: float | None = None,
) -> nn.Module:
    if hasattr(config_or_name, "model_name"):
        model_name = config_or_name.model_name
        num_classes = config_or_name.num_classes if num_classes is None else num_classes
        in_channels = config_or_name.in_channels if in_channels is None else in_channels
        dropout = config_or_name.dropout if dropout is None else dropout
    else:
        model_name = str(config_or_name)
        num_classes = 10 if num_classes is None else num_classes
        in_channels = 1 if in_channels is None else in_channels
        dropout = 0.30 if dropout is None else dropout

    normalized_name = normalize_model_name(model_name)
    if normalized_name == "small_cnn":
        return SmallCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    return MediumCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_params, trainable_params
