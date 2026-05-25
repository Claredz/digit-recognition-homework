from __future__ import annotations

import argparse
import csv
import json
import random
import struct
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from src.config import ExperimentConfig
from src.data import CorrectEMNISTOrientation
from src.error_analysis import save_error_analysis_bundle
from src.evaluate import load_model_from_checkpoint, save_evaluation_bundle
from src.experiment_config import git_commit_hash
from src.model import build_model, count_model_parameters
from src.preprocess import preprocess_to_mnist_style_image
from src.robust_data import RandomBrightnessContrast, RandomErodeDilate
from src.robust_train import freeze_backbone, unfreeze_all
from src.train import set_seed

SELECTED_MNIST_C = (
    "impulse_noise",
    "shot_noise",
    "dotted_line",
    "stripe",
    "spatter",
    "fog",
    "glass_blur",
    "motion_blur",
    "zigzag",
    "shear",
    "translate",
    "scale",
)

MNIST_C_WEIGHTS = {
    "impulse_noise": 0.12,
    "shot_noise": 0.10,
    "dotted_line": 0.10,
    "stripe": 0.10,
    "spatter": 0.10,
    "fog": 0.08,
    "glass_blur": 0.08,
    "motion_blur": 0.08,
    "zigzag": 0.08,
    "shear": 0.06,
    "translate": 0.05,
    "scale": 0.05,
}


@dataclass
class TestARobustConfig:
    project_root: Path
    output_dir: Path
    base_checkpoint: Path | None = None
    model_name: str = "medium_cnn"
    dropout: float = 0.21672530847241062
    image_size: int = 28
    seed: int = 42
    epochs: int = 8
    epoch_size: int = 180000
    batch_size: int = 2048
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.03
    num_workers: int = 8
    prefetch_factor: int = 4
    use_amp: bool = True
    allow_tf32: bool = True
    compile_model: bool = False
    log_interval: int = 20
    patience: int = 3
    clean_weight: float = 0.35
    mnist_c_weight: float = 0.40
    synthetic_weight: float = 0.25
    use_testa_partial: bool = False
    testa_weight: float = 0.0
    testa_train_ratio: float = 0.70
    use_kfold: bool = False
    kfold_n_splits: int = 5
    kfold_index: int = 0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    mix_prob: float = 0.0
    random_erasing_p: float = 0.0
    mnist_max_samples: int | None = None
    qmnist_max_samples: int | None = 60000
    emnist_max_samples: int | None = 50000
    mnist_c_train_per_corruption: int | None = 60000
    mnist_c_val_per_corruption: int = 800
    testA_weight_raw: float = 0.45
    testA_weight_preprocess: float = 0.55
    checkpoint_name: str = "robust_expert_v2_best.pt"
    training_mode: str = "mixed_robust"
    experiment_id: str = "testa_robust_v2"
    preprocessing_mode: str = "raw"
    evaluate_preprocess: bool = False
    freeze_backbone_epochs: int = 0
    affine_degrees: float = 5.0
    translate_ratio: float = 0.04
    scale_min: float = 0.96
    scale_max: float = 1.04
    shear_degrees: float = 2.0
    use_testa_like_augment: bool = False
    testa_like_augment_strength: str = "medium"
    testa_like_morph_p: float | None = None
    testa_like_blur_p: float = 0.0
    testa_like_contrast_p: float = 0.0
    testa_like_dilate_bias: float = 0.5
    preprocess_probability: float = 0.0
    save_validation_artifacts: bool = True

    def data_dir(self) -> Path:
        return self.project_root / "data"

    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    def logs_dir(self) -> Path:
        return self.output_dir / "logs"


class TensorTestALikeAugment:
    def __init__(
        self,
        strength: str = "medium",
        morph_p: float | None = None,
        blur_p: float = 0.0,
        contrast_p: float = 0.0,
        dilate_bias: float = 0.5,
    ):
        self.strength = str(strength).strip().lower()
        self.morph_gate_p = float(morph_p) if morph_p is not None else (0.10 if self.strength == "light" else 0.20)
        self.morph = RandomErodeDilate(
            p=1.0 if morph_p is not None or self.strength == "light" else 0.25,
            dilate_bias=float(dilate_bias),
        )
        self.blur_p = float(blur_p)
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))
        self.contrast_p = float(contrast_p)
        self.contrast = RandomBrightnessContrast(brightness=(0.90, 1.10), contrast=(0.85, 1.15), p=1.0)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.strength == "light":
            return self._call_light(tensor).clamp(0.0, 1.0)
        if random.random() < 0.45:
            tensor = self._gray_or_noisy_background(tensor)
        if random.random() < 0.40:
            tensor = self._salt_pepper(tensor)
        if random.random() < 0.30:
            tensor = self._occlude(tensor)
        if random.random() < 0.20:
            tensor = self._edge_cutoff(tensor)
        if self.morph_gate_p > 0 and random.random() < self.morph_gate_p:
            tensor = self.morph(tensor)
        if self.contrast_p > 0 and random.random() < self.contrast_p:
            tensor = self.contrast(tensor)
        if self.blur_p > 0 and random.random() < self.blur_p:
            tensor = self.blur(tensor)
        if random.random() < 0.12:
            tensor = 1.0 - tensor
        return tensor.clamp(0.0, 1.0)

    def _call_light(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.morph_gate_p > 0 and random.random() < self.morph_gate_p:
            tensor = self.morph(tensor)
        if self.contrast_p > 0 and random.random() < self.contrast_p:
            tensor = self.contrast(tensor)
        if self.blur_p > 0 and random.random() < self.blur_p:
            tensor = self.blur(tensor)
        return tensor

    def _gray_or_noisy_background(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor.clone()
        background = result < random.uniform(0.08, 0.22)
        gray = random.uniform(0.05, 0.45)
        noise = torch.rand_like(result) * random.uniform(0.04, 0.30)
        if random.random() < 0.45:
            stripe = torch.zeros_like(result)
            if random.random() < 0.5:
                stripe[:, :, :: random.randint(3, 7)] = random.uniform(0.10, 0.45)
            else:
                stripe[:, :: random.randint(3, 7), :] = random.uniform(0.10, 0.45)
            noise = (noise + stripe).clamp(0, 1)
        result[background] = (gray + noise[background]).clamp(0, 1)
        return torch.maximum(result, tensor)

    def _salt_pepper(self, tensor: torch.Tensor) -> torch.Tensor:
        amount = random.uniform(0.01, 0.09)
        mask = torch.rand_like(tensor)
        result = tensor.clone()
        result[mask < amount / 2] = 0.0
        result[(mask >= amount / 2) & (mask < amount)] = 1.0
        return result

    def _occlude(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor.clone()
        _, height, width = result.shape
        blocks = 1 if random.random() < 0.80 else 2
        for _ in range(blocks):
            h = random.randint(3, 10)
            w = random.randint(3, 12)
            y = random.randint(0, max(0, height - h))
            x = random.randint(0, max(0, width - w))
            value = random.choice([0.0, 1.0, random.uniform(0.20, 0.65)])
            result[:, y : y + h, x : x + w] = value
        return result

    def _edge_cutoff(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor.clone()
        side = random.choice(["top", "bottom", "left", "right"])
        amount = random.randint(1, 5)
        value = random.choice([0.0, random.uniform(0.15, 0.55)])
        if side == "top":
            result[:, :amount, :] = value
        elif side == "bottom":
            result[:, -amount:, :] = value
        elif side == "left":
            result[:, :, :amount] = value
        else:
            result[:, :, -amount:] = value
        return result


class CleanFamilyDataset(Dataset):
    def __init__(self, root: Path, image_size: int, seed: int, mnist_max: int | None, qmnist_max: int | None, emnist_max: int | None):
        transform_common = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=12, translate=(0.12, 0.12), scale=(0.82, 1.18), shear=6),
                transforms.ToTensor(),
                TensorTestALikeAugment(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        transform_emnist = transforms.Compose(
            [
                CorrectEMNISTOrientation(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.RandomAffine(degrees=12, translate=(0.12, 0.12), scale=(0.82, 1.18), shear=6),
                transforms.ToTensor(),
                TensorTestALikeAugment(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        parts: list[Dataset] = []
        specs = [
            (datasets.MNIST(root=str(root), train=True, download=True, transform=transform_common), mnist_max, seed),
            (datasets.QMNIST(root=str(root), what="train", compat=True, download=True, transform=transform_common), qmnist_max, seed + 1),
            (datasets.EMNIST(root=str(root), split="digits", train=True, download=True, transform=transform_emnist), emnist_max, seed + 2),
        ]
        for dataset, max_samples, part_seed in specs:
            if max_samples is None or max_samples >= len(dataset):
                parts.append(dataset)
            else:
                generator = torch.Generator().manual_seed(part_seed)
                indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
                parts.append(Subset(dataset, indices))
        self.parts = parts
        self.lengths = [len(part) for part in parts]
        total = sum(self.lengths)
        self.weights = [0.50, 0.30, 0.20]
        self.weights = [weight / sum(self.weights) for weight in self.weights]
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        part_index = random.choices(range(len(self.parts)), weights=self.weights, k=1)[0]
        part = self.parts[part_index]
        return part[random.randrange(len(part))]


class NpyDigitsDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        if images.ndim == 4:
            images = images.squeeze()
        self.images = images.astype(np.uint8, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class MNISTCMixtureDataset(Dataset):
    def __init__(self, root: Path, corruptions: tuple[str, ...], split: str, per_corruption: int | None, image_size: int, seed: int, augment: bool):
        self.datasets: list[Dataset] = []
        self.names: list[str] = []
        weights = []
        operations: list = [transforms.Grayscale(num_output_channels=1), transforms.Resize((image_size, image_size))]
        if augment:
            operations.append(transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.94, 1.06), shear=3))
        operations.append(transforms.ToTensor())
        if augment:
            operations.append(TensorTestALikeAugment())
        operations.append(transforms.Normalize((0.5,), (0.5,)))
        transform = transforms.Compose(operations)
        for offset, corruption in enumerate(corruptions):
            image_path = root / corruption / f"{split}_images.npy"
            label_path = root / corruption / f"{split}_labels.npy"
            if not image_path.exists() or not label_path.exists():
                continue
            images = np.load(image_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
            dataset: Dataset = NpyDigitsDataset(images, labels, transform=transform)
            if per_corruption is not None and per_corruption < len(dataset):
                generator = torch.Generator().manual_seed(seed + offset)
                indices = torch.randperm(len(dataset), generator=generator)[:per_corruption].tolist()
                dataset = Subset(dataset, indices)
            self.datasets.append(dataset)
            self.names.append(corruption)
            weights.append(MNIST_C_WEIGHTS.get(corruption, 0.05))
        if not self.datasets:
            raise FileNotFoundError(f"未找到 MNIST-C 数据: {root}")
        total_weight = sum(weights)
        self.weights = [weight / total_weight for weight in weights]
        self.total = sum(len(dataset) for dataset in self.datasets)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        dataset_index = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        dataset = self.datasets[dataset_index]
        return dataset[random.randrange(len(dataset))]


class SyntheticTestALikeDataset(Dataset):
    def __init__(self, root: Path, image_size: int, seed: int, length: int, mnist_max: int | None, qmnist_max: int | None, emnist_max: int | None):
        self.base = CleanFamilyDataset(root, image_size, seed + 100, mnist_max, qmnist_max, emnist_max)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.base[random.randrange(len(self.base))]


class WeightedMixtureDataset(Dataset):
    def __init__(self, datasets_by_name: dict[str, Dataset], weights_by_name: dict[str, float], length: int):
        self.names = list(datasets_by_name)
        self.datasets = [datasets_by_name[name] for name in self.names]
        weights = [weights_by_name[name] for name in self.names]
        total = sum(weights)
        self.weights = [weight / total for weight in weights]
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        source_index = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        dataset = self.datasets[source_index]
        return dataset[random.randrange(len(dataset))]


def stratified_indices(labels: np.ndarray, train_ratio: float, seed: int):
    generator = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    for label in range(10):
        indices = np.where(labels == label)[0]
        generator.shuffle(indices)
        split = max(1, int(round(len(indices) * train_ratio)))
        split = min(split, len(indices) - 1)
        train_indices.extend(indices[:split].tolist())
        val_indices.extend(indices[split:].tolist())
    generator.shuffle(train_indices)
    generator.shuffle(val_indices)
    return train_indices, val_indices


def kfold_indices(labels: np.ndarray, n_splits: int, fold_index: int, seed: int):
    """Stratified K-Fold by class. Returns (train_indices, val_indices) for the given fold."""
    if not 0 <= fold_index < n_splits:
        raise ValueError(f"fold_index 必须在 [0, {n_splits}) 范围内，收到 {fold_index}")
    generator = np.random.default_rng(seed)
    val_indices: list[int] = []
    train_indices: list[int] = []
    for label in range(10):
        class_indices = np.where(labels == label)[0]
        generator.shuffle(class_indices)
        folds = np.array_split(class_indices, n_splits)
        for index, fold in enumerate(folds):
            if index == fold_index:
                val_indices.extend(fold.tolist())
            else:
                train_indices.extend(fold.tolist())
    generator.shuffle(train_indices)
    generator.shuffle(val_indices)
    return train_indices, val_indices


class IdxTestADataset(Dataset):
    def __init__(self, image_path: Path, label_path: Path, preprocess: bool = False, indices: list[int] | None = None):
        images = self._read_images(image_path)
        labels = self._read_labels(label_path)
        if len(images) != len(labels):
            raise ValueError("testA images/labels 数量不一致")
        if indices is None:
            self.indices = np.arange(len(labels), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
            images = images[self.indices]
            labels = labels[self.indices]
        self.images = images
        self.labels = labels
        self.preprocess = preprocess

    @staticmethod
    def _read_images(path: Path) -> np.ndarray:
        payload = path.read_bytes()
        magic, count, rows, cols = struct.unpack(">IIII", payload[:16])
        if magic != 2051:
            raise ValueError(f"不是 IDX image 文件: {path}")
        return np.frombuffer(payload, dtype=np.uint8, offset=16).reshape(count, rows, cols)

    @staticmethod
    def _read_labels(path: Path) -> np.ndarray:
        payload = path.read_bytes()
        magic, count = struct.unpack(">II", payload[:8])
        if magic != 2049:
            raise ValueError(f"不是 IDX label 文件: {path}")
        labels = np.frombuffer(payload, dtype=np.uint8, offset=8).astype(np.int64)
        if len(labels) != count:
            raise ValueError(f"IDX label 数量不一致: {path}")
        return labels

    def __len__(self) -> int:
        return len(self.labels)

    def sample_ids(self) -> list[int]:
        return [int(index) for index in self.indices.tolist()]

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        if self.preprocess:
            image = preprocess_to_mnist_style_image(image, auto_invert=True)
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, (0.5,), (0.5,))
        return tensor, int(self.labels[index])


class TestAPartialTrainDataset(Dataset):
    def __init__(
        self,
        image_path: Path,
        label_path: Path,
        indices: list[int],
        image_size: int,
        random_erasing_p: float = 0.0,
        preprocess_probability: float = 0.50,
        use_testa_like_augment: bool = True,
        testa_like_augment_strength: str = "medium",
        testa_like_morph_p: float | None = None,
        testa_like_blur_p: float = 0.0,
        testa_like_contrast_p: float = 0.0,
        testa_like_dilate_bias: float = 0.5,
        affine_degrees: float = 5.0,
        translate_ratio: float = 0.04,
        scale_min: float = 0.96,
        scale_max: float = 1.04,
        shear_degrees: float = 2.0,
    ):
        images = IdxTestADataset._read_images(image_path)
        labels = IdxTestADataset._read_labels(label_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.images = images[self.indices]
        self.labels = labels[self.indices]
        self.preprocess_probability = preprocess_probability
        layers = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=affine_degrees,
                translate=(translate_ratio, translate_ratio),
                scale=(scale_min, scale_max),
                shear=shear_degrees,
            ),
            transforms.ToTensor(),
        ]
        if use_testa_like_augment:
            layers.append(
                TensorTestALikeAugment(
                    strength=testa_like_augment_strength,
                    morph_p=testa_like_morph_p,
                    blur_p=testa_like_blur_p,
                    contrast_p=testa_like_contrast_p,
                    dilate_bias=testa_like_dilate_bias,
                )
            )
        layers.append(transforms.Normalize((0.5,), (0.5,)))
        if random_erasing_p > 0:
            layers.append(transforms.RandomErasing(p=random_erasing_p, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0))
        self.transform = transforms.Compose(layers)

    def __len__(self) -> int:
        return len(self.labels)

    def sample_ids(self) -> list[int]:
        return [int(index) for index in self.indices.tolist()]

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        if self.preprocess_probability > 0 and random.random() < self.preprocess_probability:
            image = preprocess_to_mnist_style_image(image, auto_invert=True)
        return self.transform(image), int(self.labels[index])


def make_loader(dataset: Dataset, config: TestARobustConfig, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
        "drop_last": shuffle,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_datasets(config: TestARobustConfig):
    data_dir = config.data_dir()
    clean = CleanFamilyDataset(data_dir, config.image_size, config.seed, config.mnist_max_samples, config.qmnist_max_samples, config.emnist_max_samples)
    mnist_c_root = data_dir / "mnist_c" / "mnist_c"
    mnist_c_train = MNISTCMixtureDataset(
        mnist_c_root,
        SELECTED_MNIST_C,
        split="train",
        per_corruption=config.mnist_c_train_per_corruption,
        image_size=config.image_size,
        seed=config.seed + 10,
        augment=True,
    )
    synthetic = SyntheticTestALikeDataset(
        data_dir,
        config.image_size,
        config.seed + 20,
        length=max(1, int(config.epoch_size * config.synthetic_weight)),
        mnist_max=config.mnist_max_samples,
        qmnist_max=config.qmnist_max_samples,
        emnist_max=config.emnist_max_samples,
    )
    test_a_images = data_dir / "test_A_images.idx3-ubyte(1)" / "test_A_images.idx3-ubyte"
    test_a_labels = data_dir / "test_A_labels.idx1-ubyte(1)" / "test_A_labels.idx1-ubyte"
    all_test_a_labels = IdxTestADataset._read_labels(test_a_labels)
    test_a_train_indices: list[int] = []
    test_a_val_indices: list[int] | None = None
    train_parts: dict[str, Dataset] = {
        "clean_family": clean,
        "mnist_c_selected": mnist_c_train,
        "synthetic_testa_like": synthetic,
    }
    train_weights = {
        "clean_family": config.clean_weight,
        "mnist_c_selected": config.mnist_c_weight,
        "synthetic_testa_like": config.synthetic_weight,
    }
    if config.use_testa_partial and config.testa_weight > 0:
        if config.use_kfold:
            test_a_train_indices, test_a_val_indices = kfold_indices(all_test_a_labels, config.kfold_n_splits, config.kfold_index, config.seed + 40)
        else:
            test_a_train_indices, test_a_val_indices = stratified_indices(all_test_a_labels, config.testa_train_ratio, config.seed + 40)
        test_a_train = TestAPartialTrainDataset(
            test_a_images,
            test_a_labels,
            test_a_train_indices,
            config.image_size,
            random_erasing_p=config.random_erasing_p,
            preprocess_probability=config.preprocess_probability,
            use_testa_like_augment=config.use_testa_like_augment,
            testa_like_augment_strength=config.testa_like_augment_strength,
            testa_like_morph_p=config.testa_like_morph_p,
            testa_like_blur_p=config.testa_like_blur_p,
            testa_like_contrast_p=config.testa_like_contrast_p,
            testa_like_dilate_bias=config.testa_like_dilate_bias,
            affine_degrees=config.affine_degrees,
            translate_ratio=config.translate_ratio,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
            shear_degrees=config.shear_degrees,
        )
        train_parts["testA_partial"] = test_a_train
        train_weights["testA_partial"] = config.testa_weight
    train = WeightedMixtureDataset(train_parts, train_weights, length=config.epoch_size)
    mnist_c_val = MNISTCMixtureDataset(
        mnist_c_root,
        SELECTED_MNIST_C,
        split="test",
        per_corruption=config.mnist_c_val_per_corruption,
        image_size=config.image_size,
        seed=config.seed + 30,
        augment=False,
    )
    test_a_raw = IdxTestADataset(test_a_images, test_a_labels, preprocess=False, indices=test_a_val_indices)
    test_a_preprocess = IdxTestADataset(test_a_images, test_a_labels, preprocess=True, indices=test_a_val_indices)
    metadata = {
        "train_length_per_epoch": len(train),
        "clean_family_length": len(clean),
        "mnist_c_train_length": len(mnist_c_train),
        "synthetic_length": len(synthetic),
        "mnist_c_val_length": len(mnist_c_val),
        "testA_length": int(len(all_test_a_labels)),
        "testA_train_length": int(len(test_a_train_indices)),
        "testA_validation_length": len(test_a_raw),
        "testA_validation_mode": "kfold" if config.use_kfold else ("heldout" if test_a_val_indices is not None else "full"),
        "testA_train_ratio": config.testa_train_ratio if config.use_testa_partial else 0.0,
        "kfold_n_splits": config.kfold_n_splits if config.use_kfold else 0,
        "kfold_index": config.kfold_index if config.use_kfold else -1,
        "random_erasing_p": config.random_erasing_p,
        "mixup_alpha": config.mixup_alpha,
        "cutmix_alpha": config.cutmix_alpha,
        "mix_prob": config.mix_prob,
        "use_testa_like_augment": config.use_testa_like_augment,
        "testa_like_augment_strength": config.testa_like_augment_strength,
        "testa_like_morph_p": config.testa_like_morph_p,
        "testa_like_blur_p": config.testa_like_blur_p,
        "testa_like_contrast_p": config.testa_like_contrast_p,
        "testa_like_dilate_bias": config.testa_like_dilate_bias,
        "selected_mnist_c": list(SELECTED_MNIST_C),
        "sampling_weights": train_weights,
    }
    return train, mnist_c_val, test_a_raw, test_a_preprocess, metadata


def configure_cuda(config: TestARobustConfig):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")


def _sample_lambda(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


def _rand_bbox(height: int, width: int, lam: float) -> tuple[int, int, int, int]:
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h = int(height * cut_ratio)
    cut_w = int(width * cut_ratio)
    cy = np.random.randint(height)
    cx = np.random.randint(width)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(height, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(width, cx + cut_w // 2)
    return y1, y2, x1, x2


def maybe_mixup_cutmix(images: torch.Tensor, labels: torch.Tensor, config: TestARobustConfig):
    """Returns (images, labels_a, labels_b, lam, mode). mode in {'none','mixup','cutmix'}."""
    if config.mix_prob <= 0 or (config.mixup_alpha <= 0 and config.cutmix_alpha <= 0):
        return images, labels, labels, 1.0, "none"
    if random.random() >= config.mix_prob:
        return images, labels, labels, 1.0, "none"
    use_cutmix = config.cutmix_alpha > 0 and (config.mixup_alpha <= 0 or random.random() < 0.5)
    perm = torch.randperm(images.size(0), device=images.device)
    labels_b = labels[perm]
    if use_cutmix:
        lam = _sample_lambda(config.cutmix_alpha)
        y1, y2, x1, x2 = _rand_bbox(images.size(-2), images.size(-1), lam)
        if y2 > y1 and x2 > x1:
            images = images.clone()
            images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (images.size(-2) * images.size(-1)))
            return images, labels, labels_b, lam, "cutmix"
        return images, labels, labels, 1.0, "none"
    lam = _sample_lambda(config.mixup_alpha)
    images = lam * images + (1.0 - lam) * images[perm]
    return images, labels, labels_b, lam, "mixup"


def evaluate_loader(model, loader, device: str, use_amp: bool, desc: str):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda" and use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            pred = logits.argmax(1)
            total_loss += float(loss.item()) * images.size(0)
            total += images.size(0)
            y_true.append(labels.cpu())
            y_pred.append(pred.cpu())
    y_true_np = torch.cat(y_true).numpy()
    y_pred_np = torch.cat(y_pred).numpy()
    return {
        "name": desc,
        "loss": total_loss / max(1, total),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "num_samples": int(total),
    }


def evaluate_all(model, loaders: dict[str, DataLoader], config: TestARobustConfig, device: str):
    results = {name: evaluate_loader(model, loader, device, config.use_amp, name) for name, loader in loaders.items()}
    composite = (
        config.testA_weight_raw * results["testA_raw"]["accuracy"]
        + config.testA_weight_preprocess * results["testA_preprocess"]["accuracy"]
    )
    results["score"] = {
        "testA_composite": float(composite),
        "mnist_c_selected_accuracy": results["mnist_c_selected"]["accuracy"],
    }
    return results


def save_checkpoint(path: Path, model, config: TestARobustConfig, epoch: int, score: float, history: dict):
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "model_name": config.model_name,
        "epoch": epoch,
        "best_val_accuracy": score,
        "best_val_loss": None,
        "history": history,
    }
    torch.save(checkpoint, path)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _testa_idx_paths(config: TestARobustConfig) -> tuple[Path, Path]:
    data_dir = config.data_dir()
    return (
        data_dir / "test_A_images.idx3-ubyte(1)" / "test_A_images.idx3-ubyte",
        data_dir / "test_A_labels.idx1-ubyte(1)" / "test_A_labels.idx1-ubyte",
    )


def make_specialist_loader(dataset: Dataset, config: TestARobustConfig, shuffle: bool) -> DataLoader:
    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
        "drop_last": False,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_testa_specialist_datasets(config: TestARobustConfig):
    image_path, label_path = _testa_idx_paths(config)
    labels = IdxTestADataset._read_labels(label_path)
    split_seed = config.seed + 40
    if config.use_kfold:
        train_indices, val_indices = kfold_indices(labels, config.kfold_n_splits, config.kfold_index, split_seed)
        validation_mode = "kfold"
    else:
        train_indices, val_indices = stratified_indices(labels, config.testa_train_ratio, split_seed)
        validation_mode = "heldout"

    train_dataset = TestAPartialTrainDataset(
        image_path,
        label_path,
        train_indices,
        config.image_size,
        random_erasing_p=config.random_erasing_p,
        preprocess_probability=config.preprocess_probability,
        use_testa_like_augment=config.use_testa_like_augment,
        testa_like_augment_strength=config.testa_like_augment_strength,
        testa_like_morph_p=config.testa_like_morph_p,
        testa_like_blur_p=config.testa_like_blur_p,
        testa_like_contrast_p=config.testa_like_contrast_p,
        testa_like_dilate_bias=config.testa_like_dilate_bias,
        affine_degrees=config.affine_degrees,
        translate_ratio=config.translate_ratio,
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        shear_degrees=config.shear_degrees,
    )
    val_raw = IdxTestADataset(image_path, label_path, preprocess=False, indices=val_indices)
    val_preprocess = IdxTestADataset(image_path, label_path, preprocess=True, indices=val_indices) if config.evaluate_preprocess else None
    metadata = {
        "mode": config.training_mode,
        "dataset": "TestA-only",
        "preprocessing_mode": config.preprocessing_mode,
        "primary_metric": "testA_raw_accuracy",
        "testA_length": int(len(labels)),
        "testA_train_length": int(len(train_indices)),
        "testA_validation_length": int(len(val_indices)),
        "testA_validation_mode": validation_mode,
        "kfold_n_splits": config.kfold_n_splits if config.use_kfold else 0,
        "kfold_index": config.kfold_index if config.use_kfold else -1,
        "split_seed": split_seed,
        "augmentation": {
            "affine_degrees": config.affine_degrees,
            "translate_ratio": config.translate_ratio,
            "scale_min": config.scale_min,
            "scale_max": config.scale_max,
            "shear_degrees": config.shear_degrees,
            "mixup_alpha": config.mixup_alpha,
            "cutmix_alpha": config.cutmix_alpha,
            "mix_prob": config.mix_prob,
            "random_erasing_p": config.random_erasing_p,
            "use_testa_like_augment": config.use_testa_like_augment,
            "testa_like_augment_strength": config.testa_like_augment_strength,
            "testa_like_morph_p": config.testa_like_morph_p,
            "testa_like_blur_p": config.testa_like_blur_p,
            "testa_like_contrast_p": config.testa_like_contrast_p,
            "testa_like_dilate_bias": config.testa_like_dilate_bias,
            "preprocess_probability": config.preprocess_probability,
        },
        "train_indices": [int(index) for index in train_indices],
        "val_indices": [int(index) for index in val_indices],
    }
    return train_dataset, val_raw, val_preprocess, metadata


def _specialist_experiment_config(config: TestARobustConfig) -> ExperimentConfig:
    return ExperimentConfig(
        project_root=config.project_root,
        output_dir=config.output_dir,
        model_name=config.model_name,
        dropout=config.dropout,
        batch_size=config.batch_size,
        image_size=config.image_size,
        use_amp=config.use_amp,
        allow_tf32=config.allow_tf32,
        verbose=False,
    )


def _build_specialist_model(config: TestARobustConfig, device: str):
    exp_config = _specialist_experiment_config(config)
    if config.base_checkpoint is not None:
        checkpoint_path = Path(config.base_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到 TestA specialist 初始化 checkpoint: {checkpoint_path}")
        model, payload = load_model_from_checkpoint(checkpoint_path, exp_config, device)
        initialized_from_checkpoint = True
    else:
        model = build_model(config.model_name, num_classes=10, in_channels=1, dropout=config.dropout)
        model.to(device)
        payload = {}
        initialized_from_checkpoint = False
    return model, payload, initialized_from_checkpoint


@torch.no_grad()
def collect_probability_bundle(model, loader: DataLoader, device: str, use_amp: bool):
    model.eval()
    images_out = []
    labels_out = []
    probabilities_out = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device == "cuda" and use_amp):
            probabilities = torch.softmax(model(images), dim=1)
        images_out.append(images.cpu())
        labels_out.append(labels.cpu())
        probabilities_out.append(probabilities.cpu())
    return torch.cat(images_out), torch.cat(labels_out), torch.cat(probabilities_out)


def save_validation_prediction_artifacts(
    config: TestARobustConfig,
    model,
    val_dataset: IdxTestADataset,
    val_loader: DataLoader,
    device: str,
    preprocess_loader: DataLoader | None = None,
):
    predictions_dir = config.output_dir / "predictions"
    evaluation_dir = config.output_dir / "evaluation" / "validation_raw"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    images, labels, probabilities = collect_probability_bundle(model, val_loader, device, config.use_amp)
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    sample_ids = val_dataset.sample_ids()

    probability_path = predictions_dir / "validation_probabilities.pt"
    torch.save(
        {
            "sample_ids": sample_ids,
            "labels": labels,
            "probabilities": probabilities,
            "preprocessing_mode": "raw",
        },
        probability_path,
    )

    csv_path = predictions_dir / "validation_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "label", "prediction", "confidence", "correct"] + [f"prob_{index}" for index in range(10)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, sample_id in enumerate(sample_ids):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "label": int(labels[row_index].item()),
                    "prediction": int(predictions[row_index].item()),
                    "confidence": float(confidences[row_index].item()),
                    "correct": bool(predictions[row_index].item() == labels[row_index].item()),
                    **{f"prob_{class_index}": float(probabilities[row_index, class_index].item()) for class_index in range(10)},
                }
            )

    preprocess_probabilities = None
    if preprocess_loader is not None:
        _, preprocess_labels, preprocess_probabilities = collect_probability_bundle(model, preprocess_loader, device, config.use_amp)
        if not torch.equal(preprocess_labels, labels):
            raise RuntimeError("raw/preprocess validation label order mismatch")
        torch.save(
            {
                "sample_ids": sample_ids,
                "labels": labels,
                "probabilities": preprocess_probabilities,
                "preprocessing_mode": "preprocess",
            },
            predictions_dir / "validation_preprocess_probabilities.pt",
        )

    summary = save_evaluation_bundle(images, labels, predictions, evaluation_dir)
    error_payload = save_error_analysis_bundle(
        evaluation_dir,
        sample_ids=sample_ids,
        images=images,
        labels=labels,
        probabilities=probabilities,
        preprocess_probabilities=preprocess_probabilities,
    )
    return {
        "validation_predictions_csv": str(csv_path),
        "validation_probabilities": str(probability_path),
        "evaluation_dir": str(evaluation_dir),
        "summary": summary,
        "error_analysis": error_payload,
    }


def train_testa_specialist(config: TestARobustConfig):
    set_seed(config.seed)
    configure_cuda(config)
    config.checkpoints_dir().mkdir(parents=True, exist_ok=True)
    config.logs_dir().mkdir(parents=True, exist_ok=True)
    (config.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (config.output_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"stage": "TestA specialist", "device": device, **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}}, indent=2, ensure_ascii=False), flush=True)

    train_dataset, val_raw, val_preprocess, metadata = build_testa_specialist_datasets(config)
    write_json(config.logs_dir() / "testa_specialist_data_manifest.json", metadata)
    write_json(config.logs_dir() / "testa_split_indices.json", {"train_indices": metadata["train_indices"], "val_indices": metadata["val_indices"]})
    train_loader = make_specialist_loader(train_dataset, config, shuffle=True)
    val_batch_config = TestARobustConfig(**{**asdict(config), "batch_size": min(config.batch_size, 4096), "num_workers": max(0, min(config.num_workers, 6))})
    val_raw_loader = make_specialist_loader(val_raw, val_batch_config, shuffle=False)
    val_preprocess_loader = make_specialist_loader(val_preprocess, val_batch_config, shuffle=False) if val_preprocess is not None else None

    model, checkpoint_payload, initialized_from_checkpoint = _build_specialist_model(config, device)
    model = model.to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if config.compile_model:
        model = torch.compile(model, mode="max-autotune")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    total_params, trainable_params_initial = count_model_parameters(model)
    phase_plan: list[tuple[str, int, bool]] = []
    freeze_epochs = config.freeze_backbone_epochs if initialized_from_checkpoint else 0
    if freeze_epochs > 0:
        phase_plan.append(("freeze_backbone", min(freeze_epochs, config.epochs), True))
    remaining_epochs = max(0, config.epochs - sum(phase[1] for phase in phase_plan))
    if remaining_epochs > 0:
        phase_plan.append(("full_finetune" if initialized_from_checkpoint else "scratch", remaining_epochs, False))
    if not phase_plan:
        phase_plan.append(("full_finetune", config.epochs, False))

    history = {"epochs": [], "metadata": metadata, "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}}
    checkpoint_path = config.checkpoints_dir() / config.checkpoint_name
    csv_path = config.logs_dir() / "testa_specialist_history.csv"
    best_score = -1.0
    best_epoch = 0
    bad_epochs = 0
    global_epoch = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "phase", "train_loss", "train_acc", "testA_raw_acc", "testA_preprocess_acc", "score", "lr", "elapsed_sec"])
        writer.writeheader()
        stop_training = False
        for phase_name, phase_epochs, should_freeze in phase_plan:
            if should_freeze:
                freeze_backbone(model)
            else:
                unfreeze_all(model)
            optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, phase_epochs))
            scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and config.use_amp)

            for _ in range(phase_epochs):
                global_epoch += 1
                model.train()
                start = time.perf_counter()
                total_loss = 0.0
                total_correct = 0
                total = 0
                for batch_index, (images, labels) in enumerate(train_loader, start=1):
                    images = images.to(device, non_blocking=True)
                    if device == "cuda":
                        images = images.contiguous(memory_format=torch.channels_last)
                    labels = labels.to(device, non_blocking=True)
                    mixed_images, labels_a, labels_b, lam, mix_mode = maybe_mixup_cutmix(images, labels, config)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=device == "cuda" and config.use_amp):
                        logits = model(mixed_images)
                        if mix_mode == "none":
                            loss = criterion(logits, labels)
                        else:
                            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    pred = logits.argmax(1)
                    total_loss += float(loss.item()) * images.size(0)
                    total_correct += int((pred == labels).sum().item())
                    total += images.size(0)
                    if batch_index == 1 or batch_index % config.log_interval == 0 or batch_index == len(train_loader):
                        print(
                            f"[testa:{phase_name}] epoch {global_epoch}/{config.epochs} batch {batch_index}/{len(train_loader)} "
                            f"loss={total_loss / max(1, total):.4f} acc={total_correct / max(1, total):.4f} "
                            f"lr={optimizer.param_groups[0]['lr']:.6g} elapsed={time.perf_counter() - start:.1f}s",
                            flush=True,
                        )
                scheduler.step()
                raw_result = evaluate_loader(model, val_raw_loader, device, config.use_amp, "testA_raw")
                preprocess_acc = None
                if val_preprocess_loader is not None:
                    preprocess_acc = evaluate_loader(model, val_preprocess_loader, device, config.use_amp, "testA_preprocess")["accuracy"]
                score = raw_result["accuracy"]
                row = {
                    "epoch": global_epoch,
                    "phase": phase_name,
                    "train_loss": total_loss / max(1, total),
                    "train_acc": total_correct / max(1, total),
                    "testA_raw_acc": raw_result["accuracy"],
                    "testA_preprocess_acc": preprocess_acc,
                    "score": score,
                    "lr": optimizer.param_groups[0]["lr"],
                    "elapsed_sec": round(time.perf_counter() - start, 2),
                }
                writer.writerow(row)
                handle.flush()
                history["epochs"].append(row)
                improved = score > best_score + 1e-4
                if improved:
                    best_score = score
                    best_epoch = global_epoch
                    bad_epochs = 0
                    save_checkpoint(checkpoint_path, model, config, global_epoch, score, history)
                    status = "best"
                else:
                    bad_epochs += 1
                    status = f"no_improve={bad_epochs}"
                write_json(config.logs_dir() / "testa_specialist_history.json", history)
                print(
                    f"Epoch {global_epoch:03d}/{config.epochs:03d} | phase={phase_name} train_acc={row['train_acc']:.4f} "
                    f"testA_raw={row['testA_raw_acc']:.4f} score={score:.4f} best={best_score:.4f}@{best_epoch} | {status}",
                    flush=True,
                )
                if bad_epochs >= config.patience:
                    print(f"Early stopping: best={best_score:.4f}@{best_epoch}", flush=True)
                    stop_training = True
                    break
            if stop_training:
                break

    artifact_summary = None
    if config.save_validation_artifacts and checkpoint_path.exists():
        best_model, _ = load_model_from_checkpoint(checkpoint_path, _specialist_experiment_config(config), device)
        artifact_summary = save_validation_prediction_artifacts(
            config,
            best_model,
            val_raw,
            val_raw_loader,
            device,
            preprocess_loader=val_preprocess_loader,
        )

    _, trainable_params_final = count_model_parameters(model)
    manifest = {
        "experiment_id": config.experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(config.project_root),
        "stage": "TestA specialist",
        "training_mode": config.training_mode,
        "initialized_from_checkpoint": initialized_from_checkpoint,
        "base_checkpoint": str(config.base_checkpoint) if config.base_checkpoint else None,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(csv_path),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_score,
        "final_val_accuracy": history["epochs"][-1]["testA_raw_acc"] if history["epochs"] else None,
        "model_name": config.model_name,
        "seed": config.seed,
        "fold_index": config.kfold_index if config.use_kfold else None,
        "n_splits": config.kfold_n_splits if config.use_kfold else None,
        "preprocessing_mode": config.preprocessing_mode,
        "augmentation": metadata["augmentation"],
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "output_dir": str(config.output_dir),
        "data": {key: value for key, value in metadata.items() if key not in {"train_indices", "val_indices"}},
        "total_parameters": total_params,
        "trainable_parameters_initial": trainable_params_initial,
        "trainable_parameters_final": trainable_params_final,
        "checkpoint_payload_keys": sorted(checkpoint_payload.keys()) if isinstance(checkpoint_payload, dict) else [],
        "artifacts": artifact_summary,
    }
    write_json(config.logs_dir() / "testa_specialist_manifest.json", manifest)
    write_json(config.logs_dir() / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def train(config: TestARobustConfig):
    if config.training_mode in {"testa_scratch", "testa_specialist", "testa_finetune"}:
        return train_testa_specialist(config)
    set_seed(config.seed)
    configure_cuda(config)
    config.checkpoints_dir().mkdir(parents=True, exist_ok=True)
    config.logs_dir().mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"stage": "testA robust v2", "device": device, **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}}, indent=2, ensure_ascii=False), flush=True)

    train_dataset, mnist_c_val, test_a_raw, test_a_preprocess, metadata = build_datasets(config)
    write_json(config.logs_dir() / "testa_robust_v2_data_manifest.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)

    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_batch_config = TestARobustConfig(**{**asdict(config), "batch_size": min(config.batch_size, 4096), "num_workers": max(2, min(config.num_workers, 6))})
    val_loaders = {
        "testA_raw": make_loader(test_a_raw, val_batch_config, shuffle=False),
        "testA_preprocess": make_loader(test_a_preprocess, val_batch_config, shuffle=False),
        "mnist_c_selected": make_loader(mnist_c_val, val_batch_config, shuffle=False),
    }

    exp_config = ExperimentConfig(
        project_root=config.project_root,
        output_dir=config.output_dir,
        model_name=config.model_name,
        dropout=config.dropout,
        batch_size=config.batch_size,
        use_amp=config.use_amp,
        allow_tf32=config.allow_tf32,
        verbose=False,
    )
    model, _ = load_model_from_checkpoint(config.base_checkpoint, exp_config, device)
    model = model.to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if config.compile_model:
        model = torch.compile(model, mode="max-autotune")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and config.use_amp)

    history = {"epochs": [], "metadata": metadata, "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}}
    best_score = -1.0
    best_epoch = 0
    bad_epochs = 0
    checkpoint_path = config.checkpoints_dir() / config.checkpoint_name
    csv_path = config.logs_dir() / "testa_robust_v2_history.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_acc", "testA_raw_acc", "testA_preprocess_acc", "mnist_c_acc", "score", "lr", "elapsed_sec"])
        writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            model.train()
            start = time.perf_counter()
            total_loss = 0.0
            total_correct = 0
            total = 0
            for batch_index, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                if device == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                mixed_images, labels_a, labels_b, lam, mix_mode = maybe_mixup_cutmix(images, labels, config)
                if mix_mode != "none" and device == "cuda":
                    mixed_images = mixed_images.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device == "cuda" and config.use_amp):
                    logits = model(mixed_images)
                    if mix_mode == "none":
                        loss = criterion(logits, labels)
                    else:
                        loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                pred = logits.argmax(1)
                total_loss += float(loss.item()) * images.size(0)
                total_correct += int((pred == labels).sum().item())
                total += images.size(0)
                if batch_index == 1 or batch_index % config.log_interval == 0 or batch_index == len(train_loader):
                    print(
                        f"[train] epoch {epoch}/{config.epochs} batch {batch_index}/{len(train_loader)} "
                        f"loss={total_loss / max(1, total):.4f} acc={total_correct / max(1, total):.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6g} elapsed={time.perf_counter() - start:.1f}s",
                        flush=True,
                    )
            scheduler.step()
            eval_results = evaluate_all(model, val_loaders, config, device)
            score = eval_results["score"]["testA_composite"]
            row = {
                "epoch": epoch,
                "train_loss": total_loss / max(1, total),
                "train_acc": total_correct / max(1, total),
                "testA_raw_acc": eval_results["testA_raw"]["accuracy"],
                "testA_preprocess_acc": eval_results["testA_preprocess"]["accuracy"],
                "mnist_c_acc": eval_results["mnist_c_selected"]["accuracy"],
                "score": score,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_sec": round(time.perf_counter() - start, 2),
            }
            writer.writerow(row)
            handle.flush()
            epoch_payload = {**row, "eval": eval_results}
            history["epochs"].append(epoch_payload)
            improved = score > best_score + 1e-4
            if improved:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                save_checkpoint(checkpoint_path, model, config, epoch, score, history)
                status = "best"
            else:
                bad_epochs += 1
                status = f"no_improve={bad_epochs}"
            write_json(config.logs_dir() / "testa_robust_v2_history.json", history)
            print(
                f"Epoch {epoch:03d}/{config.epochs:03d} | train_acc={row['train_acc']:.4f} "
                f"testA_raw={row['testA_raw_acc']:.4f} testA_preprocess={row['testA_preprocess_acc']:.4f} "
                f"mnistC={row['mnist_c_acc']:.4f} score={score:.4f} best={best_score:.4f}@{best_epoch} | {status}",
                flush=True,
            )
            if bad_epochs >= config.patience:
                print(f"Early stopping: best={best_score:.4f}@{best_epoch}", flush=True)
                break
    manifest = {"best_score": best_score, "best_epoch": best_epoch, "checkpoint": str(checkpoint_path), "history_csv": str(csv_path), "data": metadata}
    write_json(config.logs_dir() / "testa_robust_v2_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="基于 robust_expert_best 的 MNIST-C + testA-like 鲁棒微调")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_submission"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("outputs_submission/checkpoints/robust_expert_best.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--epoch-size", type=int, default=180000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--clean-weight", type=float, default=0.35)
    parser.add_argument("--mnist-c-weight", type=float, default=0.40)
    parser.add_argument("--synthetic-weight", type=float, default=0.25)
    parser.add_argument("--use-testa-partial", action="store_true")
    parser.add_argument("--testa-weight", type=float, default=0.0)
    parser.add_argument("--testa-train-ratio", type=float, default=0.70)
    parser.add_argument("--use-kfold", action="store_true")
    parser.add_argument("--kfold-n-splits", type=int, default=5)
    parser.add_argument("--kfold-index", type=int, default=0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=0.0)
    parser.add_argument("--random-erasing-p", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint-name", default="robust_expert_v2_best.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TestARobustConfig(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir.resolve(),
        base_checkpoint=args.base_checkpoint.resolve(),
        epochs=args.epochs,
        epoch_size=args.epoch_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        label_smoothing=args.label_smoothing,
        clean_weight=args.clean_weight,
        mnist_c_weight=args.mnist_c_weight,
        synthetic_weight=args.synthetic_weight,
        use_testa_partial=args.use_testa_partial,
        testa_weight=args.testa_weight,
        testa_train_ratio=args.testa_train_ratio,
        use_kfold=args.use_kfold,
        kfold_n_splits=args.kfold_n_splits,
        kfold_index=args.kfold_index,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        random_erasing_p=args.random_erasing_p,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        patience=args.patience,
        compile_model=args.compile,
        checkpoint_name=args.checkpoint_name,
    )
    train(config)


if __name__ == "__main__":
    main()
