"""Evaluate one or more checkpoints on the full 3500-image TestA set.

Reports raw / preprocess accuracy with and without TTA. No ensembling.
"""

from __future__ import annotations

import argparse
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
    dataset = IdxTestADataset(image_path, label_path, preprocess=preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate_loader(model, loader, device: str, tta_n: int = 1) -> dict:
    model.eval()
    correct = 0
    total = 0
    tta_augment = transforms.RandomAffine(
        degrees=5,
        translate=(0.04, 0.04),
        scale=(0.96, 1.04),
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images), dim=1)
        for _ in range(max(0, tta_n - 1)):
            augmented = torch.stack([tta_augment(image.cpu()) for image in images]).to(device)
            probabilities = probabilities + torch.softmax(model(augmented), dim=1)
        if tta_n > 1:
            probabilities = probabilities / tta_n
        predictions = probabilities.argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += int(labels.numel())
    return {"accuracy": correct / total if total else 0.0, "num_samples": total}


def evaluate_checkpoint(name: str, checkpoint: Path, data_dir: Path, device: str, batch_size: int, tta_n: int) -> dict:
    print(f"\n=== {name} ===", flush=True)
    print(f"checkpoint: {checkpoint}", flush=True)
    config = ExperimentConfig(project_root=PROJECT_ROOT, model_name="medium_cnn", num_classes=10, in_channels=1, image_size=28)
    model, _ = load_model_from_checkpoint(checkpoint, config, device)

    image_path = data_dir / "test_A_images.idx3-ubyte(1)" / "test_A_images.idx3-ubyte"
    label_path = data_dir / "test_A_labels.idx1-ubyte(1)" / "test_A_labels.idx1-ubyte"

    raw_loader = build_loader(image_path, label_path, preprocess=False, batch_size=batch_size)
    prep_loader = build_loader(image_path, label_path, preprocess=True, batch_size=batch_size)

    results = {}
    for view_name, loader in [("raw", raw_loader), ("preprocess", prep_loader)]:
        t0 = time.perf_counter()
        baseline = evaluate_loader(model, loader, device, tta_n=1)
        baseline["elapsed_s"] = round(time.perf_counter() - t0, 2)
        print(f"  {view_name:<10s} no_tta  acc={baseline['accuracy']:.4f} n={baseline['num_samples']} t={baseline['elapsed_s']}s", flush=True)
        results[f"{view_name}_no_tta"] = baseline

        if tta_n > 1:
            t0 = time.perf_counter()
            tta_result = evaluate_loader(model, loader, device, tta_n=tta_n)
            tta_result["elapsed_s"] = round(time.perf_counter() - t0, 2)
            tta_result["tta_n"] = tta_n
            print(f"  {view_name:<10s} tta{tta_n:<2d} acc={tta_result['accuracy']:.4f} n={tta_result['num_samples']} t={tta_result['elapsed_s']}s", flush=True)
            results[f"{view_name}_tta{tta_n}"] = tta_result

    return {"checkpoint": str(checkpoint), "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs_submission" / "evaluation" / "testA_full_eval.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    summary = {"device": device, "batch_size": args.batch_size, "tta_n": args.tta_n, "checkpoints": {}}
    for item in args.checkpoints:
        if "=" not in item:
            raise ValueError(f"--checkpoints expects name=path, got: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        summary["checkpoints"][name] = evaluate_checkpoint(name, path, args.data_dir, device, args.batch_size, args.tta_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {args.output}", flush=True)

    print("\n=== summary (accuracy on full TestA, n=3500) ===")
    header = f"{'ckpt':<22s} {'raw':>8s} {'raw+tta':>10s} {'prep':>8s} {'prep+tta':>10s}"
    print(header)
    for name, payload in summary["checkpoints"].items():
        r = payload["results"]
        raw = r["raw_no_tta"]["accuracy"]
        raw_tta = r.get(f"raw_tta{args.tta_n}", {}).get("accuracy")
        prep = r["preprocess_no_tta"]["accuracy"]
        prep_tta = r.get(f"preprocess_tta{args.tta_n}", {}).get("accuracy")
        raw_tta_s = f"{raw_tta:.4f}" if raw_tta is not None else "-"
        prep_tta_s = f"{prep_tta:.4f}" if prep_tta is not None else "-"
        print(f"{name:<22s} {raw:.4f}   {raw_tta_s:>8s}   {prep:.4f}   {prep_tta_s:>8s}")


if __name__ == "__main__":
    main()
