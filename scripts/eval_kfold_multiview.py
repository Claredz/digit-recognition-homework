"""Multi-view fusion ensemble for K-Fold + plan_A on full TestA.

For each model and view (raw / preprocess) we compute:
- baseline probabilities (no TTA)
- TTA probabilities (n views averaged)

Then we report several fusion strategies:
- raw-only / prep-only / raw+prep
- baseline-only / baseline+TTA
- 5-fold vs 5-fold+plan_A (6 members)

Saves a JSON breakdown and prints a sorted table.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.evaluate import load_model_from_checkpoint
from src.testa_robust_train import IdxTestADataset


def build_loader(image_path: Path, label_path: Path, preprocess: bool, batch_size: int) -> DataLoader:
    return DataLoader(IdxTestADataset(image_path, label_path, preprocess=preprocess),
                      batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def get_probs(model, loader, device: str, tta_n: int) -> torch.Tensor:
    model.eval()
    tta_augment = transforms.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.96, 1.04),
                                          interpolation=transforms.InterpolationMode.BILINEAR, fill=-1.0)
    chunks = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1)
        for _ in range(max(0, tta_n - 1)):
            augmented = torch.stack([tta_augment(image.cpu()) for image in images]).to(device)
            probs = probs + torch.softmax(model(augmented), dim=1)
        if tta_n > 1:
            probs = probs / tta_n
        chunks.append(probs.cpu())
    return torch.cat(chunks)


def accuracy_of(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return float((probs.argmax(dim=1) == labels).float().mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs_submission" / "evaluation" / "testA_kfold_multiview.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} tta_n: {args.tta_n}", flush=True)

    config = ExperimentConfig(project_root=PROJECT_ROOT, model_name="medium_cnn", num_classes=10, in_channels=1, image_size=28)
    image_path = args.data_dir / "test_A_images.idx3-ubyte(1)" / "test_A_images.idx3-ubyte"
    label_path = args.data_dir / "test_A_labels.idx1-ubyte(1)" / "test_A_labels.idx1-ubyte"

    raw_loader = build_loader(image_path, label_path, preprocess=False, batch_size=args.batch_size)
    prep_loader = build_loader(image_path, label_path, preprocess=True, batch_size=args.batch_size)
    import numpy as np
    labels = torch.from_numpy(IdxTestADataset._read_labels(label_path)).long()

    checkpoints: dict[str, Path] = {}
    for item in args.checkpoints:
        if "=" not in item:
            raise ValueError(f"--checkpoints expects name=path, got: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        checkpoints[name] = path

    per_model_views: dict[str, dict[str, torch.Tensor]] = {}
    per_model_acc: dict[str, dict[str, float]] = {}
    for name, path in checkpoints.items():
        t0 = time.perf_counter()
        print(f"\n[{name}] loading {path.name}", flush=True)
        model, _ = load_model_from_checkpoint(path, config, device)
        views = {
            "raw":      get_probs(model, raw_loader, device, tta_n=1),
            "raw_tta":  get_probs(model, raw_loader, device, tta_n=args.tta_n),
            "prep":     get_probs(model, prep_loader, device, tta_n=1),
            "prep_tta": get_probs(model, prep_loader, device, tta_n=args.tta_n),
        }
        per_model_views[name] = views
        per_model_acc[name] = {k: accuracy_of(v, labels) for k, v in views.items()}
        print(f"[{name}] " + " ".join(f"{k}={v:.4f}" for k, v in per_model_acc[name].items())
              + f" t={time.perf_counter()-t0:.1f}s", flush=True)

    # Helper to ensemble a list of probability tensors (mean)
    def ensemble(probs_list: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(probs_list, dim=0).mean(dim=0)

    # Build cross-model fusions for several view-combinations and member-sets
    member_sets = {"kfold_5": [n for n in checkpoints if "kfold" in n],
                   "kfold_5_plus_planA": list(checkpoints.keys())}
    view_combos = {
        "raw_only":         ["raw"],
        "raw+tta":          ["raw", "raw_tta"],
        "prep_only":        ["prep"],
        "prep+tta":         ["prep", "prep_tta"],
        "raw+prep":         ["raw", "prep"],
        "raw+prep+tta":     ["raw", "raw_tta", "prep", "prep_tta"],
    }
    fusion_table = []
    for set_name, members in member_sets.items():
        for view_name, view_keys in view_combos.items():
            # Per-model: average across selected views; then mean across models
            per_model_probs = [ensemble([per_model_views[m][k] for k in view_keys]) for m in members]
            fused = ensemble(per_model_probs)
            accuracy = accuracy_of(fused, labels)
            fusion_table.append({
                "members": set_name,
                "n_members": len(members),
                "views": view_name,
                "accuracy": accuracy,
            })

    fusion_table.sort(key=lambda row: row["accuracy"], reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "device": device,
        "tta_n": args.tta_n,
        "per_model": per_model_acc,
        "fusions": fusion_table,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== per-model accuracy ===")
    print(f"{'model':<14s} {'raw':>8s} {'raw_tta':>9s} {'prep':>8s} {'prep_tta':>10s}")
    for name, acc in per_model_acc.items():
        print(f"{name:<14s} {acc['raw']:.4f}   {acc['raw_tta']:.4f}    {acc['prep']:.4f}    {acc['prep_tta']:.4f}")

    print("\n=== fusion results (sorted by accuracy, n=3500) ===")
    print(f"{'members':<22s} {'views':<18s} {'acc':>8s}")
    for row in fusion_table:
        print(f"{row['members']:<22s} {row['views']:<18s} {row['accuracy']:.4f}")

    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
