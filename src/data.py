"""数据集适配与 DataLoader 构建。

这个模块对不同数据源提供一个很小的抽象层，让其余代码只依赖统一接口。

支持的数据集：
- MNIST（torchvision.datasets.MNIST）
- 文件夹数字数据集（按类别子目录组织）
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
    """从文件夹结构读取数字图片的数据集。

    期望目录结构：

        root/
          0/  # 类别目录名必须能转成 int
            img1.png
            ...
          1/
            ...

    为保证可复现性：先对类别目录排序，再对每个类别里的文件路径排序。
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
            raise ValueError(f"在目录 {self.root} 下没有找到带标签的图片")

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
        f"不支持的 dataset_name='{config.dataset_name}'。请使用 'mnist' 或 'folder'。"
    )


def create_dataloaders(config: ExperimentConfig):
    dataset = load_base_dataset(config)

    val_size = max(1, int(len(dataset) * config.validation_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("validation_split 设置过大，导致训练样本数为 0")

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
