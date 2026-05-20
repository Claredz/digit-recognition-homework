"""Evaluate K-Fold ensemble on full TestA (3500 images).

For each fold checkpoint:
- Predict on full TestA raw + preprocess views.
- Optionally also compute per-fold heldout accuracy (the 700 images that fold did not see).

Then ensemble by averaging probabilities across folds.

Outputs JSON summary and prints a comparison table including:
- per-fold accuracy on full TestA
- per-fold heldout accuracy (clean estimate)
- ensemble accuracy on full TestA
- ensemble heldout accuracy (mean across folds, each fold's heldout chunk only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.evaluate import load_model_from_checkpoint
from src.testa_robust_train import IdxTestADataset, kfold_indices


def build_loader(image_path: Path, label_path: Path, preprocess: bool, batch_size: int) -> DataLoader:
    dataset = IdxTestADataset(image_path, label_path, preprocess=preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def get_probabilities(model, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_probs = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images), dim=1)
        all_probs.append(probabilities.cpu())
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)


def evaluate_per_fold(checkpoints: dict[str, Path], data_dir: Path, batch_size: int, device: str, n_splits: int, seed: int) -> dict:
    config = ExperimentConfig(project_root=PROJECT_ROOT, model_name="medium_cnn", num_classes=10, in_channels=1, image_size=28)
    image_path = data_dir / "test_A_images.idx3-ubyte(1)" / "test_A_images.idx3-ubyte"
    label_path = data_dir / "test_A_labels.idx1-ubyte(1)" / "test_A_labels.idx1-ubyte"
    raw_loader = build_loader(image_path, label_path, preprocess=False, batch_size=batch_size)
    prep_loader = build_loader(image_path, label_path, preprocess=True, batch_size=batch_size)
    all_labels_array = IdxTestADataset._read_labels(label_path)

    fold_records = {}
    raw_prob_stack = []
    prep_prob_stack = []
    held_correct_raw = 0
    held_correct_prep = 0
    held_total = 0

    for name, path in checkpoints.items():
        fold_index = int(name.split("_f")[-1].split("_")[0]) if "_f" in name else -1
        print(f"\n[fold {fold_index}] loading {path}", flush=True)
        model, _ = load_model_from_checkpoint(path, config, device)

        t0 = time.perf_counter()
        raw_probs, labels = get_probabilities(model, raw_loader, device)
        prep_probs, _ = get_probabilities(model, prep_loader, device)
        elapsed = round(time.perf_counter() - t0, 2)

        raw_predictions = raw_probs.argmax(dim=1)
        prep_predictions = prep_probs.argmax(dim=1)
        raw_full_accuracy = float((raw_predictions == labels).float().mean().item())
        prep_full_accuracy = float((prep_predictions == labels).float().mean().item())

        heldout_summary = None
        if fold_index >= 0 and 0 <= fold_index < n_splits:
            _, val_idx = kfold_indices(all_labels_array, n_splits, fold_index, seed + 40)
            val_idx_tensor = torch.tensor(val_idx, dtype=torch.long)
            raw_held = float((raw_predictions[val_idx_tensor] == labels[val_idx_tensor]).float().mean().item())
            prep_held = float((prep_predictions[val_idx_tensor] == labels[val_idx_tensor]).float().mean().item())
            held_correct_raw += int((raw_predictions[val_idx_tensor] == labels[val_idx_tensor]).sum().item())
            held_correct_prep += int((prep_predictions[val_idx_tensor] == labels[val_idx_tensor]).sum().item())
            held_total += int(val_idx_tensor.numel())
            heldout_summary = {"raw": raw_held, "preprocess": prep_held, "n": int(val_idx_tensor.numel())}

        print(f"[fold {fold_index}] full raw={raw_full_accuracy:.4f} prep={prep_full_accuracy:.4f} heldout={heldout_summary} elapsed={elapsed}s", flush=True)
        fold_records[name] = {
            "fold_index": fold_index,
            "checkpoint": str(path),
            "full_testa": {"raw": raw_full_accuracy, "preprocess": prep_full_accuracy, "n": int(labels.numel())},
            "heldout": heldout_summary,
            "elapsed_sec": elapsed,
        }
        raw_prob_stack.append(raw_probs)
        prep_prob_stack.append(prep_probs)

    raw_ensemble = torch.stack(raw_prob_stack, dim=0).mean(dim=0)
    prep_ensemble = torch.stack(prep_prob_stack, dim=0).mean(dim=0)
    labels = torch.from_numpy(all_labels_array).long()
    raw_ensemble_accuracy = float((raw_ensemble.argmax(dim=1) == labels).float().mean().item())
    prep_ensemble_accuracy = float((prep_ensemble.argmax(dim=1) == labels).float().mean().item())
    ensemble_summary = {
        "full_testa_raw": raw_ensemble_accuracy,
        "full_testa_preprocess": prep_ensemble_accuracy,
        "full_testa_n": int(labels.numel()),
        "mean_heldout_raw": (held_correct_raw / held_total) if held_total else None,
        "mean_heldout_preprocess": (held_correct_prep / held_total) if held_total else None,
        "heldout_n_total": held_total,
    }
    return {"folds": fold_records, "ensemble": ensemble_summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs (name should include _fN for heldout calc)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs_submission" / "evaluation" / "testA_kfold_ensemble.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    checkpoints: dict[str, Path] = {}
    for item in args.checkpoints:
        if "=" not in item:
            raise ValueError(f"--checkpoints expects name=path, got: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        checkpoints[name] = path

    payload = evaluate_per_fold(checkpoints, args.data_dir, args.batch_size, device, args.n_splits, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Per-fold accuracy ===")
    print(f"{'fold':<10s} {'full_raw':>10s} {'full_prep':>10s} {'held_raw':>10s} {'held_prep':>10s}")
    for name, record in payload["folds"].items():
        full = record["full_testa"]
        held = record["heldout"]
        held_raw = f"{held['raw']:.4f}" if held else "-"
        held_prep = f"{held['preprocess']:.4f}" if held else "-"
        print(f"{name:<10s} {full['raw']:.4f}     {full['preprocess']:.4f}     {held_raw:>8s}    {held_prep:>8s}")

    e = payload["ensemble"]
    print("\n=== Ensemble (probability mean) ===")
    print(f"full TestA raw       : {e['full_testa_raw']:.4f}  (n={e['full_testa_n']})")
    print(f"full TestA preprocess: {e['full_testa_preprocess']:.4f}")
    if e["mean_heldout_raw"] is not None:
        print(f"mean heldout raw     : {e['mean_heldout_raw']:.4f}  (n={e['heldout_n_total']}, each sample evaluated by its own fold)")
        print(f"mean heldout prep    : {e['mean_heldout_preprocess']:.4f}")
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
