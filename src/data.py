"""Dataset adapters and dataloaders.

This module provides a small abstraction around the supported datasets so the
rest of the codebase can depend on a consistent interface.

Supported datasets:
- MNIST via torchvision.datasets.MNIST
- Folder-based digit dataset with class subdirectories
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from src.config import ExperimentConfig


class FolderDigitsDataset(Dataset[tuple[torch.Tensor, int]]):
    """A dataset that reads digit images from a folder structure.

    Expected layout:

        root/
          0/  # class label directory name must be an int
            img1.png
            ...
          1/
            ...

    Ordering is deterministic by sorting class directories then sorting image
    file paths within each class directory.
    """

    def __init__(self, root: Path, image_size: int = 28, augment: bool = False):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.augment = bool(augment)

        self.samples: list[tuple[Path, int]] = []

        class_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        for class_dir in class_dirs:
            label = int(class_dir.name)
            for image_path in sorted(class_dir.glob("*")):
                if image_path.is_file():
                    self.samples.append((image_path, label))

        self.transform = _build_transform(image_size=self.image_size, augment=self.augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("L")
            tensor = self.transform(image)

        return tensor, label


def _build_transform(image_size: int, augment: bool) -> transforms.Compose:
    transform_list: list[transforms.Transform] = [
        transforms.Resize((image_size, image_size)),
    ]

    if augment:
        transform_list.extend(
            [
                transforms.RandomRotation(8),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    return transforms.Compose(transform_list)


def load_base_dataset(config: ExperimentConfig, *, train: bool = True, augment: bool = False) -> Dataset:
    """Load a base dataset based on configuration."""

    dataset_name = (config.dataset_name or "").lower().strip()
    data_dir = config.resolved_data_dir()

    if dataset_name == "mnist":
        transform = _build_transform(image_size=config.image_size, augment=augment)
        return datasets.MNIST(root=str(data_dir), train=train, download=True, transform=transform)

    if dataset_name == "folder":
        if config.data_dir is None:
            raise ValueError("data_dir must be provided when dataset_name='folder'")
        return FolderDigitsDataset(Path(config.data_dir), image_size=config.image_size, augment=augment)

    raise ValueError(f"Unsupported dataset_name: {config.dataset_name!r}")


def create_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders with deterministic splitting."""

    dataset = load_base_dataset(config, train=True, augment=False)

    val_size = max(1, int(len(dataset) * config.validation_split))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader
