"""Evaluate TestA-specialist experts on standard benchmarks (MNIST/EMNIST/QMNIST)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import build_eval_transform
from src.evaluate import load_model_from_checkpoint

CHECKPOINTS = [
    # (label, checkpoint_path, model_name)
    (
        "wide_resnet_tiny (best expert)",
        PROJECT_ROOT / "outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/seed_42/fold_0/checkpoints/testa_specialist_best.pt",
        "wide_resnet_tiny",
    ),
    (
        "medium_v2_anti1_margin_seed42",
        PROJECT_ROOT / "outputs_runs/testa_medium_v2_anti1_margin_seed42_e60/seed_42/fold_0/checkpoints/testa_specialist_best.pt",
        "medium_cnn",
    ),
    (
        "medium_v2_anti1_margin_seed2026",
        PROJECT_ROOT / "outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/seed_2026/fold_0/checkpoints/testa_specialist_best.pt",
        "medium_cnn",
    ),
    (
        "medium_v2_raw_seed3407",
        PROJECT_ROOT / "outputs_runs/testa_medium_v2_raw_seed3407_e60/seed_3407/fold_0/checkpoints/testa_specialist_best.pt",
        "medium_cnn",
    ),
    (
        "clean_model (outputs_submission)",
        PROJECT_ROOT / "outputs_submission/checkpoints/best_model_state.pt",
        "medium_cnn",
    ),
]

HOLDOUTS = [
    ("mnist_test", lambda cfg: datasets.MNIST(root=str(cfg.resolved_data_dir()), train=False, download=True, transform=build_eval_transform(cfg))),
    ("emnist_digits_test", lambda cfg: datasets.EMNIST(root=str(cfg.resolved_data_dir()), split="digits", train=False, download=True, transform=build_eval_transform(cfg, correct_emnist=True))),
    ("qmnist_test10k", lambda cfg: datasets.QMNIST(root=str(cfg.resolved_data_dir()), what="test10k", compat=True, download=True, transform=build_eval_transform(cfg))),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ExperimentConfig(project_root=PROJECT_ROOT)

    results: list[dict] = []

    for label, ckpt_path, model_name in CHECKPOINTS:
        if not ckpt_path.exists():
            print(f"[SKIP] {label}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[EVAL] {label}")
        print(f"  checkpoint: {ckpt_path}")
        print(f"  device: {device}")

        config.model_name = model_name
        model, _ = load_model_from_checkpoint(ckpt_path, config, device)
        model.eval()

        row = {"label": label}
        for holdout_name, ds_fn in HOLDOUTS:
            dataset = ds_fn(config)
            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in loader:
                    logits = model(images.to(device))
                    preds = logits.argmax(dim=1).cpu()
                    all_preds.append(preds)
                    all_labels.append(labels.cpu())

            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_labels).numpy()
            acc = float(accuracy_score(y_true, y_pred))
            row[holdout_name] = acc
            print(f"  {holdout_name}: {acc:.4f} (n={len(y_true)})")

        results.append(row)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Model':<40} {'MNIST':>8} {'EMNIST':>8} {'QMNIST':>8} {'Avg':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        avg = (r.get("mnist_test", 0) + r.get("emnist_digits_test", 0) + r.get("qmnist_test10k", 0)) / 3
        print(
            f"{r['label']:<40} "
            f"{r.get('mnist_test', 0):8.4f} "
            f"{r.get('emnist_digits_test', 0):8.4f} "
            f"{r.get('qmnist_test10k', 0):8.4f} "
            f"{avg:8.4f}"
        )

    out_path = PROJECT_ROOT / "outputs_runs/expert_standard_benchmark_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
