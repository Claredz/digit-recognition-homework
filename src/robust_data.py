from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

from src.config import ExperimentConfig
from src.data import CorrectEMNISTOrientation, _dataset, _loader_kwargs, _source_specs, _split_indices, build_eval_transform
from src.preprocess import preprocess_to_mnist_style_image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class AddGaussianNoise:
    def __init__(self, std_range=(0.02, 0.08), p: float = 0.35):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        std = random.uniform(*self.std_range)
        return (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)


class SaltPepperNoise:
    def __init__(self, amount_range=(0.005, 0.03), p: float = 0.20):
        self.amount_range = amount_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        amount = random.uniform(*self.amount_range)
        mask = torch.rand_like(tensor)
        result = tensor.clone()
        result[mask < amount / 2] = 0.0
        result[(mask >= amount / 2) & (mask < amount)] = 1.0
        return result


class RandomBrightnessContrast:
    def __init__(self, brightness=(0.70, 1.30), contrast=(0.65, 1.50), p: float = 0.40):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        brightness = random.uniform(*self.brightness)
        contrast = random.uniform(*self.contrast)
        mean = tensor.mean(dim=(-2, -1), keepdim=True)
        return ((tensor - mean) * contrast + mean * brightness).clamp(0.0, 1.0)


class RandomErodeDilate:
    def __init__(self, p: float = 0.35, dilate_bias: float = 0.5):
        self.p = p
        self.dilate_bias = float(dilate_bias)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        kernel = random.choice([2, 3])
        batch = tensor.unsqueeze(0)
        if random.random() < self.dilate_bias:
            morphed = F.max_pool2d(batch, kernel_size=kernel, stride=1, padding=kernel // 2)
        else:
            morphed = -F.max_pool2d(-batch, kernel_size=kernel, stride=1, padding=kernel // 2)
        morphed = morphed.squeeze(0)[..., : tensor.shape[-2], : tensor.shape[-1]]
        if morphed.max() <= 0.03:
            return tensor
        return morphed.clamp(0.0, 1.0)


class RandomBackgroundNoise:
    def __init__(self, p: float = 0.25, strength=(0.01, 0.08)):
        self.p = p
        self.strength = strength

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        strength = random.uniform(*self.strength)
        noise = torch.rand_like(tensor) * strength
        background = tensor < 0.20
        result = tensor.clone()
        result[background] = (result[background] + noise[background]).clamp(0.0, 1.0)
        return result


class CenterJitter:
    def __init__(self, max_shift: int = 2, p: float = 0.30):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        shift_y = random.randint(-self.max_shift, self.max_shift)
        shift_x = random.randint(-self.max_shift, self.max_shift)
        result = torch.zeros_like(tensor)
        height, width = tensor.shape[-2:]
        src_y0 = max(0, -shift_y)
        src_y1 = min(height, height - shift_y)
        src_x0 = max(0, -shift_x)
        src_x1 = min(width, width - shift_x)
        dst_y0 = max(0, shift_y)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, shift_x)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        if src_y1 > src_y0 and src_x1 > src_x0:
            result[..., dst_y0:dst_y1, dst_x0:dst_x1] = tensor[..., src_y0:src_y1, src_x0:src_x1]
        return result


class RandomInvert:
    def __init__(self, p: float = 0.04):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        return 1.0 - tensor


@dataclass
class SourceDataset:
    name: str
    train_dataset: Dataset
    val_dataset: Dataset | None
    train_length: int


class PreprocessedFolderDigitsDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform=None,
        image_size: int = 28,
        auto_invert: bool = True,
        cache_images: bool = False,
    ):
        self.root = Path(root)
        self.transform = transform
        self.image_size = image_size
        self.auto_invert = auto_invert
        self.cache_images = cache_images
        self.samples: list[tuple[Path, int, str | None]] = []
        self.cached_images: list[Image.Image] | None = None
        self._collect_samples()
        if not self.samples:
            raise ValueError(f"在目录 {self.root} 下没有找到 0-9 标签图片")
        if self.cache_images:
            self.cached_images = [self._load_preprocessed_image(path) for path, _, _ in self.samples]

    def _collect_samples(self):
        if not self.root.exists():
            return
        direct_label_dirs = [path for path in self.root.iterdir() if path.is_dir() and path.name.isdigit()]
        if direct_label_dirs:
            for class_dir in sorted(direct_label_dirs):
                label = int(class_dir.name)
                if 0 <= label <= 9:
                    self._add_images(class_dir, label, writer_id=None)
            return

        for writer_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            for class_dir in sorted(path for path in writer_dir.iterdir() if path.is_dir() and path.name.isdigit()):
                label = int(class_dir.name)
                if 0 <= label <= 9:
                    self._add_images(class_dir, label, writer_id=writer_dir.name)

    def _add_images(self, class_dir: Path, label: int, writer_id: str | None):
        for image_path in sorted(class_dir.glob("*")):
            if image_path.is_file() and image_path.suffix.lower() in _IMAGE_SUFFIXES:
                self.samples.append((image_path, label, writer_id))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_preprocessed_image(self, image_path: Path):
        with Image.open(image_path) as image:
            return preprocess_to_mnist_style_image(
                image,
                image_size=self.image_size,
                auto_invert=self.auto_invert,
            ).copy()

    def __getitem__(self, index: int):
        image_path, label, _ = self.samples[index]
        if self.cached_images is None:
            processed = self._load_preprocessed_image(image_path)
        else:
            processed = self.cached_images[index]
        if self.transform is not None:
            processed = self.transform(processed)
        return processed, label


def build_robust_train_transform(config: ExperimentConfig, correct_emnist: bool = False):
    strength = str(config.robust_aug_strength).strip().lower()
    if strength == "strong":
        translate, scale, shear = 0.18, (0.70, 1.30), 10
    elif strength == "light":
        translate, scale, shear = 0.12, (0.85, 1.15), 6
    else:
        translate, scale, shear = 0.16, (0.75, 1.25), 8

    operations: list = []
    if correct_emnist:
        operations.append(CorrectEMNISTOrientation())
    operations.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomAffine(
                degrees=config.robust_affine_degrees,
                translate=(translate, translate),
                scale=scale,
                shear=shear,
            ),
            transforms.ToTensor(),
            RandomBrightnessContrast(),
            AddGaussianNoise(std_range=(config.robust_noise_std_min, config.robust_noise_std_max)),
            SaltPepperNoise(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=config.robust_blur_prob),
            RandomErodeDilate(p=config.robust_morph_prob),
            RandomBackgroundNoise(),
            CenterJitter(max_shift=config.robust_center_jitter),
            RandomInvert(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transforms.Compose(operations)


def build_robust_eval_transform(config: ExperimentConfig, correct_emnist: bool = False):
    return build_eval_transform(config, correct_emnist=correct_emnist)


def _enabled_path(config: ExperimentConfig, attr: str, default_name: str) -> Path:
    value = getattr(config, attr)
    if value is not None:
        return Path(value)
    return config.resolved_data_dir() / default_name


def _maybe_folder_dataset(name: str, enabled: bool, root: Path, transform, config: ExperimentConfig):
    if not enabled:
        return None
    if not root.exists():
        warnings.warn(f"{name} 数据目录不存在，已跳过: {root}", stacklevel=2)
        return None
    try:
        return PreprocessedFolderDigitsDataset(
            root,
            transform=transform,
            image_size=config.image_size,
            cache_images=config.cache_folder_digits,
        )
    except ValueError as exc:
        warnings.warn(f"{name} 数据不可用，已跳过: {exc}", stacklevel=2)
        return None


def maybe_load_local_digits(config: ExperimentConfig, transform):
    return _maybe_folder_dataset(
        "local_digits",
        config.use_local_digits,
        _enabled_path(config, "local_digits_dir", "local_digits"),
        transform,
        config,
    )


def maybe_load_hasyv2_digits(config: ExperimentConfig, transform):
    return _maybe_folder_dataset("HASYv2 digits", config.use_hasyv2, _enabled_path(config, "hasyv2_dir", "hasyv2_digits"), transform, config)


def maybe_load_chars74k_digits(config: ExperimentConfig, transform):
    return _maybe_folder_dataset(
        "Chars74K EnglishHnd digits",
        config.use_chars74k,
        _enabled_path(config, "chars74k_dir", "chars74k_digits"),
        transform,
        config,
    )


def maybe_load_penbased_digits(config: ExperimentConfig, transform):
    return _maybe_folder_dataset(
        "UCI Pen-Based rendered",
        config.use_penbased_rendered,
        _enabled_path(config, "penbased_dir", "penbased_rendered"),
        transform,
        config,
    )


def maybe_load_optical_digits(config: ExperimentConfig, transform):
    return _maybe_folder_dataset(
        "UCI Optical digits",
        config.use_optical_digits,
        _enabled_path(config, "optical_dir", "optical_digits"),
        transform,
        config,
    )


def _split_optional_dataset(dataset: PreprocessedFolderDigitsDataset, validation_split: float, seed: int):
    writer_ids = [writer for _, _, writer in dataset.samples]
    if any(writer_ids) and len(set(writer_ids)) > 1:
        writers = sorted({writer for writer in writer_ids if writer is not None})
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(writers), generator=generator).tolist()
        shuffled = [writers[index] for index in order]
        val_writer_count = max(1, int(len(shuffled) * validation_split))
        val_writers = set(shuffled[:val_writer_count])
        train_indices = [index for index, writer in enumerate(writer_ids) if writer not in val_writers]
        val_indices = [index for index, writer in enumerate(writer_ids) if writer in val_writers]
        if train_indices and val_indices:
            return train_indices, val_indices
    return _split_indices(len(dataset), validation_split, seed=seed)


def _make_weighted_sampler(lengths: list[int], names: list[str], config: ExperimentConfig):
    configured = {
        "mnist_family": config.mnist_family_weight,
        "local_digits": config.local_digits_weight,
        "hasyv2": config.hasyv2_weight,
        "chars74k": config.chars74k_weight,
        "penbased": config.penbased_weight,
        "optical": config.optical_weight,
    }
    source_weights = []
    for name in names:
        source_weights.append(configured.get(name, 0.05))
    total_weight = sum(source_weights) or 1.0
    sample_weights = []
    for length, source_weight in zip(lengths, source_weights):
        sample_weights.extend([source_weight / total_weight / max(1, length)] * length)
    return WeightedRandomSampler(sample_weights, num_samples=sum(lengths), replacement=True)


def create_robust_finetune_dataloaders(config: ExperimentConfig):
    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []
    source_lengths: list[int] = []
    source_names: list[str] = []

    mnist_train_parts = []
    mnist_val_parts = []
    for offset, (source, max_samples) in enumerate(_source_specs(config)):
        correct_emnist = source == "emnist_digits"
        train_dataset = _dataset(source, config, train=True, transform=build_robust_train_transform(config, correct_emnist))
        eval_dataset = _dataset(source, config, train=True, transform=build_robust_eval_transform(config, correct_emnist))
        train_indices, val_indices = _split_indices(
            len(eval_dataset),
            config.validation_split,
            seed=config.seed + offset,
            max_samples=max_samples,
        )
        mnist_train_parts.append(Subset(train_dataset, train_indices))
        mnist_val_parts.append(Subset(eval_dataset, val_indices))

    mnist_train = mnist_train_parts[0] if len(mnist_train_parts) == 1 else ConcatDataset(mnist_train_parts)
    mnist_val = mnist_val_parts[0] if len(mnist_val_parts) == 1 else ConcatDataset(mnist_val_parts)
    train_parts.append(mnist_train)
    val_parts.append(mnist_val)
    source_lengths.append(len(mnist_train))
    source_names.append("mnist_family")

    optional_specs = [
        ("local_digits", maybe_load_local_digits),
        ("hasyv2", maybe_load_hasyv2_digits),
        ("chars74k", maybe_load_chars74k_digits),
        ("penbased", maybe_load_penbased_digits),
        ("optical", maybe_load_optical_digits),
    ]
    robust_transform = build_robust_train_transform(config)
    eval_transform = build_robust_eval_transform(config)
    for index, (name, loader) in enumerate(optional_specs, start=10):
        dataset = loader(config, robust_transform)
        eval_dataset = loader(config, eval_transform)
        if dataset is None or eval_dataset is None:
            continue
        train_indices, val_indices = _split_optional_dataset(eval_dataset, config.validation_split, seed=config.seed + index)
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(eval_dataset, val_indices)
        train_parts.append(train_subset)
        val_parts.append(val_subset)
        source_lengths.append(len(train_subset))
        source_names.append(name)

    train_dataset = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_dataset = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)

    train_kwargs = _loader_kwargs(config, shuffle=not config.use_robust_sampler)
    if config.use_robust_sampler:
        train_kwargs.pop("shuffle", None)
        train_kwargs["sampler"] = _make_weighted_sampler(source_lengths, source_names, config)
    val_kwargs = _loader_kwargs(config, shuffle=False)
    metadata = {"source_names": source_names, "source_lengths": dict(zip(source_names, source_lengths))}
    return DataLoader(train_dataset, **train_kwargs), DataLoader(val_dataset, **val_kwargs), metadata
