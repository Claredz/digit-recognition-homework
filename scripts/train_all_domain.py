#!/usr/bin/env python
"""All-domain generalist training + multi-domain evaluation.

Trains a model on Clean Family + external domains, then evaluates on each
domain separately and on the held-out TestA set via OOF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms

_proj = Path(__file__).resolve().parents[1]
if str(_proj) not in sys.path:
    sys.path.insert(0, str(_proj))

from src.model import build_model


def _tfm():
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def _load_mnist_family(data_dir: Path) -> dict[str, Dataset]:
    from src.testa_robust_train import CleanFamilyDataset

    ds: dict[str, Dataset] = {}
    try:
        clean = CleanFamilyDataset(
            data_dir, image_size=28, seed=42,
            mnist_max=None, qmnist_max=60000, emnist_max=50000,
        )
        ds["clean_family"] = clean
        print(f"  clean_family: {len(clean)} samples (MNIST+QMNIST+EMNIST)")
    except Exception as e:
        print(f"  clean_family: failed - {e}")
    return ds


def _load_image_folder(path: Path) -> Dataset | None:
    imgs = sorted(list(path.rglob("*.png")) + list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg")))
    if not imgs:
        return None
    samples: list[tuple[torch.Tensor, int]] = []
    tfm = _tfm()
    for p in imgs:
        try:
            label_str = p.parent.name
            label = int(label_str)
        except (ValueError, TypeError):
            continue
        from PIL import Image
        try:
            img = Image.open(p).convert("L")
            samples.append((tfm(img), label))
        except Exception:
            pass
    if not samples:
        return None
    return _ListDataset(samples)


class _ListDataset(Dataset):
    def __init__(self, data: list[tuple[torch.Tensor, int]]):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def build_all_domain_datasets(data_dir: Path) -> dict[str, Dataset]:
    datasets_dict = _load_mnist_family(data_dir)

    folder_domains = {
        "penbased": data_dir / "penbased_rendered",
        "hasyv2": data_dir / "hasyv2_digits",
        "chars74k": data_dir / "chars74k_digits",
    }
    for name, p in folder_domains.items():
        ds = _load_image_folder(p)
        if ds is not None:
            datasets_dict[name] = ds

    return datasets_dict


def train_one_epoch(model, loader, optimizer, device, use_amp, scaler):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_domain(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += len(labels)
    return correct / max(1, total)


@torch.no_grad()
def evaluate_testa_oof(model, device, project_root: Path):
    from src.testa_robust_train import IdxTestADataset

    imgs_path = project_root / "data" / "test_A_images.idx3-ubyte(1)" / "test_A_images.idx3-ubyte"
    lbls_path = project_root / "data" / "test_A_labels.idx1-ubyte(1)" / "test_A_labels.idx1-ubyte"
    ds = IdxTestADataset(imgs_path, lbls_path, preprocess=False)
    images_np = ds.images
    labels_np = ds.labels

    all_probs = []
    all_labels = []
    bs = 256
    model.eval()
    for start in range(0, len(labels_np), bs):
        end = min(start + bs, len(labels_np))
        batch = torch.from_numpy(images_np[start:end]).float().unsqueeze(1) / 255.0
        batch = (batch - 0.5) / 0.5
        lbls = torch.from_numpy(labels_np[start:end])
        with torch.no_grad():
            logits = model(batch.to(device))
            probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu())
        all_labels.append(lbls)

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = probs.argmax(-1)
    acc = (preds == labels).float().mean().item()
    return acc, probs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium_cnn")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_runs/all_domain_generalist"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    project_root = args.project_root.resolve()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    print("[all_domain] Loading domains...")
    domain_ds = build_all_domain_datasets(project_root / "data")
    print(f"[all_domain] Loaded {len(domain_ds)} domains: {list(domain_ds.keys())}")

    # Create unified dataset with domain labels
    all_samples: list[tuple[torch.Tensor, int]] = []
    domain_counts: dict[str, int] = {}
    for name, ds in domain_ds.items():
        n = 0
        for i in range(len(ds)):
            img, lbl = ds[i]
            all_samples.append((img, lbl))
            n += 1
        domain_counts[name] = n
        print(f"  {name}: {n} samples")

    unified = _ListDataset(all_samples)
    print(f"[all_domain] Total: {len(unified)} samples")

    # Balanced sampling weights
    total = sum(domain_counts.values())
    sample_weights = []
    offset = 0
    for name, n in domain_counts.items():
        weight = total / max(1, len(domain_counts)) / max(1, n)
        sample_weights.extend([weight] * n)
    sampler = WeightedRandomSampler(sample_weights, num_samples=min(len(unified), 200000), replacement=True)
    loader = DataLoader(unified, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)

    # Build model
    model = build_model(args.model, num_classes=10, in_channels=1, dropout=0.2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    results: dict[str, Any] = {
        "model": args.model,
        "epochs": args.epochs,
        "domains": list(domain_counts.keys()),
        "domain_counts": domain_counts,
        "per_epoch": [],
    }

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, True, scaler)
        scheduler.step()

        # Evaluate on each domain
        domain_accs: dict[str, float] = {}
        for name, ds in domain_ds.items():
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            domain_accs[name] = evaluate_domain(model, dl, device)

        # Evaluate on TestA OOF
        testa_acc, _, _ = evaluate_testa_oof(model, device, project_root)

        record = {"epoch": epoch, "loss": round(loss, 4), "testA": round(testa_acc, 4)}
        record.update({f"dom_{k}": round(v, 4) for k, v in domain_accs.items()})
        results["per_epoch"].append(record)
        print(f"Epoch {epoch:03d} loss={loss:.4f} testA={testa_acc:.4f} | " + " ".join(f"{k}={v:.3f}" for k, v in domain_accs.items()))

    # Save results
    results["best_testa"] = max(r["testA"] for r in results["per_epoch"])
    best_epoch = max(results["per_epoch"], key=lambda r: r["testA"])
    results["best_epoch"] = best_epoch["epoch"]

    out_path = output_dir / "all_domain_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[all_domain] Best TestA OOF: {results['best_testa']:.4f} @ epoch {results['best_epoch']}")
    print(f"[all_domain] Results: {out_path}")


if __name__ == "__main__":
    main()
