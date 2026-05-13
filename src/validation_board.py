from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from src.config import ExperimentConfig
from src.data import create_dataloaders
from src.evaluate import _holdout_dataset
from src.robust_data import (
    maybe_load_chars74k_digits,
    maybe_load_hasyv2_digits,
    maybe_load_local_digits,
    maybe_load_penbased_digits,
)


class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if torch.is_tensor(image):
            image = (image * 0.5 + 0.5).clamp(0, 1)
        return self.transform(image), label


def build_corrupt_lite_transform(config: ExperimentConfig):
    return transforms.Compose(
        [
            transforms.RandomAffine(degrees=12, translate=(0.12, 0.12), scale=(0.82, 1.18), shear=6),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.20),
            transforms.RandomAdjustSharpness(sharpness_factor=0.6, p=0.15),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def _loader(dataset: Dataset, config: ExperimentConfig):
    return DataLoader(
        dataset,
        batch_size=config.external_validation_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def build_validation_board_loaders(config: ExperimentConfig):
    if config.verbose:
        print("[validation-board] building loaders", flush=True)
    loaders = {}
    _, clean_val_loader = create_dataloaders(config)
    loaders["val_clean"] = clean_val_loader
    if config.verbose:
        print(f"[validation-board] add val_clean samples={len(clean_val_loader.dataset)}", flush=True)

    try:
        corrupt_dataset = TransformDataset(clean_val_loader.dataset, build_corrupt_lite_transform(config))
        loaders["val_corrupt_lite"] = _loader(corrupt_dataset, config)
        if config.verbose:
            print(f"[validation-board] add val_corrupt_lite samples={len(corrupt_dataset)}", flush=True)
    except Exception as exc:
        warnings.warn(f"Val-corrupt-lite 构建失败，已跳过: {exc}", stacklevel=2)

    external_parts = []
    for name in config.external_holdout_names:
        try:
            external_parts.append(_holdout_dataset(name, config))
        except Exception as exc:
            warnings.warn(f"外部 holdout {name} 不可用，已跳过: {exc}", stacklevel=2)
    if external_parts:
        external_dataset = external_parts[0] if len(external_parts) == 1 else ConcatDataset(external_parts)
        loaders["val_external"] = _loader(external_dataset, config)
        if config.verbose:
            print(f"[validation-board] add val_external holdouts samples={len(external_dataset)}", flush=True)

    local_holdout = config.local_digits_holdout_dir
    if local_holdout is not None:
        original_local_dir = config.local_digits_dir
        original_use_local = config.use_local_digits
        config.local_digits_dir = Path(local_holdout)
        config.use_local_digits = True
        local_dataset = maybe_load_local_digits(config, transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
        config.local_digits_dir = original_local_dir
        config.use_local_digits = original_use_local
        if local_dataset is not None:
            loaders["val_local"] = _loader(local_dataset, config)
            if config.verbose:
                print(f"[validation-board] add val_local samples={len(local_dataset)}", flush=True)

    optional_external_parts = []
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    for loader_fn in [maybe_load_hasyv2_digits, maybe_load_chars74k_digits, maybe_load_penbased_digits]:
        dataset = loader_fn(config, eval_transform)
        if dataset is not None:
            optional_external_parts.append(dataset)
    if optional_external_parts:
        existing = loaders.get("val_external")
        datasets = optional_external_parts if existing is None else [existing.dataset, *optional_external_parts]
        external_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        loaders["val_external"] = _loader(external_dataset, config)
        if config.verbose:
            print(f"[validation-board] add optional external data; val_external samples={len(external_dataset)}", flush=True)

    return loaders


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def evaluate_loader(model, loader, device: str, config: ExperimentConfig | None = None, split_name: str = "split"):
    model.eval()
    total_correct = 0
    total_examples = 0
    verbose = True if config is None else config.verbose
    log_interval = max(1, 50 if config is None else config.log_interval)
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_examples += int(labels.numel())
            if verbose and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches):
                total_text = str(total_batches) if total_batches is not None else "?"
                print(f"[validation-board:{split_name}] batch {batch_index}/{total_text}", flush=True)
    return {"accuracy": total_correct / total_examples if total_examples else 0.0, "num_samples": total_examples}


def score_validation_board(results: dict, config: ExperimentConfig | None = None):
    has_local = "val_local" in results and results["val_local"].get("num_samples", 0) > 0
    if has_local:
        weights = {"val_clean": 0.45, "val_local": 0.35, "val_external": 0.15, "val_corrupt_lite": 0.05}
    else:
        weights = {"val_clean": 0.60, "val_external": 0.25, "val_corrupt_lite": 0.15}
    if config is not None:
        overrides = {
            "val_clean": config.validation_weight_clean,
            "val_external": config.validation_weight_external,
            "val_corrupt_lite": config.validation_weight_corrupt_lite,
            "val_local": config.validation_weight_local,
        }
        weights.update({name: value for name, value in overrides.items() if value is not None})
    available = {name: weight for name, weight in weights.items() if name in results}
    total_weight = sum(available.values()) or 1.0
    composite = sum(results[name]["accuracy"] * weight for name, weight in available.items()) / total_weight
    return {"composite_score": composite, "weights": available, "has_local": has_local}


def save_validation_board(results: dict, output_dir: Path, prefix: str, config: ExperimentConfig | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = {"splits": results, "score": score_validation_board(results, config=config)}
    (output_dir / f"validation_board_{prefix}.json").write_text(
        json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / f"validation_board_{prefix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "accuracy", "num_samples"])
        writer.writeheader()
        for name, row in results.items():
            writer.writerow({"split": name, **row})
    return scored


def evaluate_validation_board(model, config: ExperimentConfig, output_dir: Path, device: str, prefix: str = "model"):
    loaders = build_validation_board_loaders(config)
    results = {}
    for name, loader in loaders.items():
        if config.verbose:
            print(f"[validation-board] evaluating {prefix}:{name}", flush=True)
        results[name] = evaluate_loader(model, loader, device, config=config, split_name=name)
        if config.verbose:
            row = results[name]
            print(f"[validation-board] done {prefix}:{name} accuracy={row['accuracy']:.4f} samples={row['num_samples']}", flush=True)
    scored = save_validation_board(results, output_dir, prefix=prefix, config=config)
    if config.verbose:
        print(f"[validation-board] {prefix} composite_score={scored['score']['composite_score']:.4f}", flush=True)
    return scored
