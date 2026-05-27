"""Evaluate the final weighted ensemble across four project domains."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import build_eval_transform
from src.evaluate import load_model_from_checkpoint

EXPERTS = [
    {
        "label": "wide_resnet_tiny_raw_seed42",
        "model_name": "wide_resnet_tiny",
        "weight": 0.7,
        "run": "testa_wide_resnet_tiny_raw_seed42_e60",
        "seed": 42,
        "oof": PROJECT_ROOT / "outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_anti1_seed2026",
        "model_name": "medium_cnn",
        "weight": 0.2,
        "run": "testa_medium_v2_anti1_margin_seed2026_e60",
        "seed": 2026,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_raw_seed3407",
        "model_name": "medium_cnn",
        "weight": 0.1,
        "run": "testa_medium_v2_raw_seed3407_e60",
        "seed": 3407,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_raw_seed3407_e60/oof/oof_probabilities.pt",
    },
]


def expert_fold_paths(expert: dict) -> list[Path]:
    return [
        PROJECT_ROOT / f"outputs_runs/{expert['run']}/seed_{expert['seed']}/fold_{fold}/checkpoints/testa_specialist_best.pt"
        for fold in range(5)
    ]


class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L")
            if self.transform is not None:
                image = self.transform(image)
        return image, target


def load_experts(config: ExperimentConfig, device: str):
    loaded = []
    for expert in EXPERTS:
        fold_models = []
        config.model_name = expert["model_name"]
        for checkpoint in expert_fold_paths(expert):
            model, _ = load_model_from_checkpoint(checkpoint, config, device)
            model.eval()
            fold_models.append(model)
        loaded.append({**expert, "models": fold_models})
    return loaded


def ensemble_predict_batch(experts: list[dict], images: torch.Tensor, device: str) -> torch.Tensor:
    images = images.to(device)
    probs = None
    with torch.no_grad():
        for expert in experts:
            fold_probs = torch.stack([torch.softmax(model(images), dim=1) for model in expert["models"]]).mean(dim=0)
            weighted = float(expert["weight"]) * fold_probs
            probs = weighted if probs is None else probs + weighted
    return probs.argmax(dim=1).cpu()


def eval_loader(experts: list[dict], loader: DataLoader, device: str) -> dict:
    preds = []
    labels = []
    for images, batch_labels in loader:
        preds.append(ensemble_predict_batch(experts, images, device))
        labels.append(batch_labels.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "num_samples": int(len(y_true))}


def weighted_group_accuracy(rows: list[dict]) -> float:
    total = sum(row["num_samples"] for row in rows)
    return sum(row["accuracy"] * row["num_samples"] for row in rows) / total


def eval_testa_oof() -> dict:
    combined = None
    labels = None
    sample_ids = None
    for expert in EXPERTS:
        payload = torch.load(expert["oof"], map_location="cpu")
        if labels is None:
            labels = payload["labels"]
            sample_ids = payload["sample_ids"]
        else:
            if sample_ids != payload["sample_ids"] or not torch.equal(labels, payload["labels"]):
                raise ValueError(f"OOF sample order mismatch for {expert['label']}")
        weighted = float(expert["weight"]) * payload["probabilities"]
        combined = weighted if combined is None else combined + weighted
    predictions = combined.argmax(dim=1)
    return {"accuracy": float((predictions == labels).float().mean().item()), "num_samples": int(len(labels))}


def eval_mnist_c(experts: list[dict], config: ExperimentConfig, device: str) -> dict:
    zip_path = config.resolved_data_dir() / "mnist_c" / "mnist_c.zip"
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        corruptions = sorted(
            {
                parts[1]
                for item in archive.namelist()
                if (parts := item.split("/")) and len(parts) == 3 and parts[2] == "test_images.npy"
            }
        )
        for corruption in corruptions:
            with archive.open(f"mnist_c/{corruption}/test_images.npy") as image_file:
                images_np = np.load(image_file)
            with archive.open(f"mnist_c/{corruption}/test_labels.npy") as label_file:
                labels_np = np.load(label_file)
            if images_np.ndim == 4 and images_np.shape[-1] == 1:
                images_np = np.squeeze(images_np, axis=-1)
            images = torch.from_numpy(images_np).float().unsqueeze(1) / 255.0
            images = (images - 0.5) / 0.5
            labels = torch.from_numpy(labels_np).long()
            loader = DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=config.external_validation_batch_size, shuffle=False)
            row = eval_loader(experts, loader, device)
            row["dataset"] = corruption
            rows.append(row)
            print(f"  MNIST-C/{corruption}: {row['accuracy']:.4f}")
    return {"accuracy": weighted_group_accuracy(rows), "num_samples": sum(row["num_samples"] for row in rows), "children": rows}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ExperimentConfig(project_root=PROJECT_ROOT, verbose=False, batch_size=512, external_validation_batch_size=512)
    experts = load_experts(config, device)
    eval_transform = build_eval_transform(config)

    groups: dict[str, dict] = {}

    mnist_family_specs = [
        ("MNIST test", datasets.MNIST(root=str(config.resolved_data_dir()), train=False, download=True, transform=eval_transform)),
        ("EMNIST Digits test", datasets.EMNIST(root=str(config.resolved_data_dir()), split="digits", train=False, download=True, transform=build_eval_transform(config, correct_emnist=True))),
        ("QMNIST test10k", datasets.QMNIST(root=str(config.resolved_data_dir()), what="test10k", compat=True, download=True, transform=eval_transform)),
        ("USPS train (local)", datasets.USPS(root=str(config.resolved_data_dir()), train=True, download=False, transform=eval_transform)),
    ]
    rows = []
    print("[group] MNIST-family")
    for name, dataset in mnist_family_specs:
        row = eval_loader(experts, DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0), device)
        row["dataset"] = name
        rows.append(row)
        print(f"  {name}: {row['accuracy']:.4f} n={row['num_samples']}")
    groups["MNIST-family"] = {"accuracy": weighted_group_accuracy(rows), "num_samples": sum(r["num_samples"] for r in rows), "children": rows}

    external_specs = [
        ("penbased_rendered", PROJECT_ROOT / "data/penbased_rendered"),
        ("hasyv2_digits", PROJECT_ROOT / "data/hasyv2_digits"),
        ("chars74k_digits", PROJECT_ROOT / "data/chars74k_digits"),
    ]
    rows = []
    print("[group] local/external digits")
    for name, path in external_specs:
        dataset = GrayscaleImageFolder(str(path), transform=eval_transform)
        row = eval_loader(experts, DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0), device)
        row["dataset"] = name
        rows.append(row)
        print(f"  {name}: {row['accuracy']:.4f} n={row['num_samples']}")
    groups["local/external digits"] = {"accuracy": weighted_group_accuracy(rows), "num_samples": sum(r["num_samples"] for r in rows), "children": rows}

    print("[group] TestA")
    groups["TestA"] = eval_testa_oof()
    print(f"  TestA OOF: {groups['TestA']['accuracy']:.4f} n={groups['TestA']['num_samples']}")

    print("[group] MNIST-C")
    groups["MNIST-C"] = eval_mnist_c(experts, config, device)
    print(f"  MNIST-C mean: {groups['MNIST-C']['accuracy']:.4f} n={groups['MNIST-C']['num_samples']}")

    output = {
        "ensemble": [{"label": e["label"], "weight": e["weight"], "n_folds": 5} for e in EXPERTS],
        "groups": groups,
    }
    out_path = PROJECT_ROOT / "outputs_runs/final_weighted_ensemble_four_domain_eval.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
