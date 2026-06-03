"""Evaluate final weighted TestA ensemble on standard benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets

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
        "folds": [
            PROJECT_ROOT / f"outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/seed_42/fold_{i}/checkpoints/testa_specialist_best.pt"
            for i in range(5)
        ],
    },
    {
        "label": "medium_anti1_seed2026",
        "model_name": "medium_cnn",
        "weight": 0.2,
        "folds": [
            PROJECT_ROOT / f"outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/seed_2026/fold_{i}/checkpoints/testa_specialist_best.pt"
            for i in range(5)
        ],
    },
    {
        "label": "medium_raw_seed3407",
        "model_name": "medium_cnn",
        "weight": 0.1,
        "folds": [
            PROJECT_ROOT / f"outputs_runs/testa_medium_v2_raw_seed3407_e60/seed_3407/fold_{i}/checkpoints/testa_specialist_best.pt"
            for i in range(5)
        ],
    },
]

HOLDOUTS = [
    ("mnist_test", lambda cfg: datasets.MNIST(root=str(cfg.resolved_data_dir()), train=False, download=True, transform=build_eval_transform(cfg))),
    ("emnist_digits_test", lambda cfg: datasets.EMNIST(root=str(cfg.resolved_data_dir()), split="digits", train=False, download=True, transform=build_eval_transform(cfg, correct_emnist=True))),
    ("qmnist_test10k", lambda cfg: datasets.QMNIST(root=str(cfg.resolved_data_dir()), what="test10k", compat=True, download=True, transform=build_eval_transform(cfg))),
]


def load_expert_models(config: ExperimentConfig, device: str):
    loaded = []
    for expert in EXPERTS:
        config.model_name = expert["model_name"]
        fold_models = []
        for checkpoint in expert["folds"]:
            model, _ = load_model_from_checkpoint(checkpoint, config, device)
            model.eval()
            fold_models.append(model)
        loaded.append({**expert, "models": fold_models})
    return loaded


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ExperimentConfig(project_root=PROJECT_ROOT, verbose=False)
    experts = load_expert_models(config, device)
    rows = []

    for holdout_name, ds_fn in HOLDOUTS:
        dataset = ds_fn(config)
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        preds_all = []
        labels_all = []
        print(f"[eval] {holdout_name} n={len(dataset)}")
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                weighted_probs = None
                for expert in experts:
                    fold_probs = []
                    for model in expert["models"]:
                        fold_probs.append(torch.softmax(model(images), dim=1))
                    expert_probs = torch.stack(fold_probs).mean(dim=0)
                    contribution = float(expert["weight"]) * expert_probs
                    weighted_probs = contribution if weighted_probs is None else weighted_probs + contribution
                preds_all.append(weighted_probs.argmax(dim=1).cpu())
                labels_all.append(labels.cpu())
        y_pred = torch.cat(preds_all).numpy()
        y_true = torch.cat(labels_all).numpy()
        acc = float(accuracy_score(y_true, y_pred))
        rows.append({"domain": holdout_name, "accuracy": acc, "num_samples": int(len(y_true))})
        print(f"  accuracy={acc:.4f}")

    out = {
        "ensemble": [{"label": e["label"], "weight": e["weight"], "n_folds": len(e["folds"])} for e in EXPERTS],
        "standard_holdouts": rows,
        "standard_average": sum(r["accuracy"] for r in rows) / len(rows),
        "testa_oof_accuracy": 0.7891,
    }
    out_path = PROJECT_ROOT / "outputs_runs/final_weighted_ensemble_standard_benchmarks.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
