"""Dataset adapters and dataloaders.

This module provides a small abstraction around the supported datasets so the
rest of the codebase can depend on a consistent interface.

Supported datasets:
- MNIST via torchvision.datasets.MNIST
- Folder-based digit dataset with class subdirectories
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from src.config import ExperimentConfig


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class FolderDigitsDataset(Dataset):
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
        self.transform = build_transform(image_size=image_size, augment=augment)
        self.samples: list[tuple[Path, int]] = []

        class_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        for class_dir in class_dirs:
            label = int(class_dir.name)
            for image_path in sorted(class_dir.glob("*")):
                if image_path.is_file() and image_path.suffix.lower() in _IMAGE_SUFFIXES:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No labeled images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            tensor = self.transform(image)
        return tensor, label


def build_transform(image_size: int, augment: bool = False):
    operations: list = [transforms.Resize((image_size, image_size))]
    if augment:
        operations.extend(
            [
                transforms.RandomRotation(8),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transforms.Compose(operations)


def load_base_dataset(config: ExperimentConfig):
    if config.dataset_name == "mnist":
        return datasets.MNIST(
            root=str(config.resolved_data_dir()),
            train=True,
            download=True,
            transform=build_transform(config.image_size, augment=False),
        )
    if config.dataset_name == "folder":
        return FolderDigitsDataset(
            root=config.resolved_data_dir(),
            image_size=config.image_size,
            augment=False,
        )
    raise ValueError(
        f"Unsupported dataset_name='{config.dataset_name}'. Use 'mnist' or 'folder'."
    )


def create_dataloaders(config: ExperimentConfig):
    dataset = load_base_dataset(config)

    val_size = max(1, int(len(dataset) * config.validation_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("validation_split leaves no training samples")

    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader
