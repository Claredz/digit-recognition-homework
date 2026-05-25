from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from src.config import ExperimentConfig

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class ToTensorIfNeeded:
    def __call__(self, image):
        if torch.is_tensor(image):
            return image.float()
        return TF.to_tensor(image)


class FolderDigitsDataset(Dataset):
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


class CorrectEMNISTOrientation:
    def __call__(self, image):
        return TF.hflip(TF.rotate(image, -90))


def _base_transform_ops(image_size: int, correct_emnist: bool = False):
    operations: list = []
    if correct_emnist:
        operations.append(CorrectEMNISTOrientation())
    operations.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
        ]
    )
    return operations


def build_transform(image_size: int, augment: bool = False):
    operations = _base_transform_ops(image_size)
    if augment:
        operations.extend(
            [
                transforms.RandomRotation(8),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            ]
        )
    operations.extend([ToTensorIfNeeded(), transforms.Normalize((0.5,), (0.5,))])
    return transforms.Compose(operations)


def build_train_transform(config: ExperimentConfig, correct_emnist: bool = False):
    operations = _base_transform_ops(config.image_size, correct_emnist=correct_emnist)
    if config.use_random_affine:
        operations.append(
            transforms.RandomAffine(
                degrees=config.rotation_degrees,
                translate=(config.translate_ratio, config.translate_ratio),
                scale=(config.scale_min, config.scale_max),
                shear=config.shear_degrees,
            )
        )
    if config.use_gaussian_blur:
        operations.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25))
    operations.extend([ToTensorIfNeeded(), transforms.Normalize((0.5,), (0.5,))])
    return transforms.Compose(operations)


def build_eval_transform(config: ExperimentConfig, correct_emnist: bool = False):
    operations = _base_transform_ops(config.image_size, correct_emnist=correct_emnist)
    operations.extend([ToTensorIfNeeded(), transforms.Normalize((0.5,), (0.5,))])
    return transforms.Compose(operations)


def _split_indices(length: int, validation_split: float, seed: int, max_samples: int | None = None):
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    if max_samples is not None:
        indices = indices[: min(max_samples, len(indices))]
    val_size = max(1, int(len(indices) * validation_split))
    train_size = len(indices) - val_size
    if train_size <= 0:
        raise ValueError("validation_split 设置过大，导致训练样本数为 0")
    return indices[:train_size], indices[train_size:]


def _dataset(source: str, config: ExperimentConfig, train: bool, transform):
    root = str(config.resolved_data_dir())
    if source == "mnist":
        return datasets.MNIST(root=root, train=train, download=True, transform=transform)
    if source == "emnist_digits":
        return datasets.EMNIST(root=root, split="digits", train=train, download=True, transform=transform)
    if source == "usps":
        return datasets.USPS(root=root, train=train, download=True, transform=transform)
    if source == "qmnist":
        return datasets.QMNIST(
            root=root,
            what="train" if train else "test10k",
            compat=True,
            download=True,
            transform=transform,
        )
    raise ValueError(f"不支持的数据源: {source}")


def _source_specs(config: ExperimentConfig):
    specs: list[tuple[str, int | None]] = []
    if config.use_mnist:
        specs.append(("mnist", config.mnist_max_samples or config.max_samples))
    if config.use_emnist_digits:
        specs.append(("emnist_digits", config.emnist_max_samples or config.max_samples))
    if config.use_usps:
        specs.append(("usps", config.usps_max_samples or config.max_samples))
    if config.use_qmnist:
        specs.append(("qmnist", config.qmnist_max_samples or config.max_samples))
    if not specs:
        raise ValueError("至少需要启用一个训练数据源")
    return specs


def _loader_kwargs(config: ExperimentConfig, shuffle: bool):
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "timeout": config.dataloader_timeout,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


def load_base_dataset(config: ExperimentConfig, train: bool = True, augment: bool = False):
    if config.dataset_name == "mnist":
        return datasets.MNIST(
            root=str(config.resolved_data_dir()),
            train=train,
            download=True,
            transform=build_train_transform(config) if augment else build_eval_transform(config),
        )
    if config.dataset_name == "folder":
        return FolderDigitsDataset(
            root=config.resolved_data_dir(),
            image_size=config.image_size,
            augment=augment,
        )
    raise ValueError(
        f"不支持的 dataset_name='{config.dataset_name}'。请使用 'mnist'、'folder' 或 'multisource'。"
    )


def create_multisource_dataloaders(config: ExperimentConfig):
    train_parts = []
    val_parts = []
    for offset, (source, max_samples) in enumerate(_source_specs(config)):
        correct_emnist = source == "emnist_digits"
        train_dataset = _dataset(source, config, train=True, transform=build_train_transform(config, correct_emnist))
        eval_dataset = _dataset(source, config, train=True, transform=build_eval_transform(config, correct_emnist))
        train_indices, val_indices = _split_indices(
            len(eval_dataset),
            config.validation_split,
            seed=config.seed + offset,
            max_samples=max_samples,
        )
        train_parts.append(Subset(train_dataset, train_indices))
        val_parts.append(Subset(eval_dataset, val_indices))

    train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_dataset = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    return (
        DataLoader(train_dataset, **_loader_kwargs(config, shuffle=True)),
        DataLoader(val_dataset, **_loader_kwargs(config, shuffle=False)),
    )


def create_dataloaders(config: ExperimentConfig):
    if config.dataset_name in {"multisource", "submission"} or any(
        [config.use_emnist_digits, config.use_usps, config.use_qmnist]
    ):
        return create_multisource_dataloaders(config)

    train_dataset = load_base_dataset(config, train=True, augment=True)
    val_dataset = load_base_dataset(config, train=True, augment=False)
    train_indices, val_indices = _split_indices(
        len(val_dataset),
        config.validation_split,
        seed=config.seed,
        max_samples=config.max_samples,
    )
    return (
        DataLoader(Subset(train_dataset, train_indices), **_loader_kwargs(config, shuffle=True)),
        DataLoader(Subset(val_dataset, val_indices), **_loader_kwargs(config, shuffle=False)),
    )
