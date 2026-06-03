"""Evaluate 5-expert dynamic MoE submission on other datasets.

Fast diagnostic: TTA=1, standard holdouts and optional small subsets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = PROJECT_ROOT / "build" / "dynamic_moe_5expert_submission"
sys.path.insert(0, str(SUBMISSION))

# Import from generated self-contained predict.py
from predict import EXPERTS, compute_dynamic_weights, compute_rule_weights, load_expert_model, router_weights

OUT = PROJECT_ROOT / "outputs_runs" / "moe_dynamic_router" / "other_domains_5expert.json"


def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def eval_transform_emnist():
    def correct(img):
        return transforms.functional.hflip(transforms.functional.rotate(img, -90))
    return transforms.Compose([correct, transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


@torch.no_grad()
def predict_expert_dataset(expert: dict, loader: DataLoader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    expert_probs = []
    labels_out = []
    if expert["kind"] == "kfold":
        fold_sum = None
        for fold in range(5):
            model = load_expert_model(SUBMISSION, expert, fold, device)
            probs_batches = []
            labels_batches = []
            for images, labels in loader:
                images = images.to(device)
                probs_batches.append(torch.softmax(model(images), dim=1).cpu())
                labels_batches.append(labels.cpu())
            fold_probs = torch.cat(probs_batches, dim=0)
            if fold_sum is None:
                labels_out = labels_batches
                fold_sum = fold_probs
            else:
                fold_sum += fold_probs
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        return fold_sum / 5.0, torch.cat(labels_out, dim=0)

    model = load_expert_model(SUBMISSION, expert, None, device)
    probs_batches = []
    labels_batches = []
    for images, labels in loader:
        images = images.to(device)
        probs_batches.append(torch.softmax(model(images), dim=1).cpu())
        labels_batches.append(labels.cpu())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(probs_batches, dim=0), torch.cat(labels_batches, dim=0)


def metrics(final_probs: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> dict:
    pred = final_probs.argmax(dim=1)
    acc = float((pred == labels).float().mean().item())
    class_acc = []
    for cls in range(10):
        mask = labels == cls
        class_acc.append(float((pred[mask] == labels[mask]).float().mean().item()) if mask.any() else None)
    out = {
        "accuracy": acc,
        "n_samples": int(labels.numel()),
        "mean_confidence": float(final_probs.max(dim=1).values.mean().item()),
        "class_1_overprediction_ratio": float((pred == 1).sum().item() / max(1, (labels == 1).sum().item())),
        "class_8_accuracy": class_acc[8],
        "x_to_1_errors": int(((pred == 1) & (labels != 1)).sum().item()),
        "per_class_accuracy": class_acc,
    }
    if weights is not None:
        out["mean_weights"] = [float(x) for x in weights.mean(dim=0).tolist()]
        out["weight_std"] = [float(x) for x in weights.std(dim=0).tolist()]
    return out


def evaluate_domain(name: str, dataset, device: str, batch_size: int = 512) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    expert_probs = []
    labels_ref = None
    print(f"\n[domain] {name} n={len(dataset)}")
    for expert in EXPERTS:
        print(f"  [expert] {expert['name']}", flush=True)
        probs, labels = predict_expert_dataset(expert, loader, device)
        expert_probs.append(probs.unsqueeze(1))
        labels_ref = labels if labels_ref is None else labels_ref
    probs = torch.cat(expert_probs, dim=1)
    labels = labels_ref
    result = {"domain": name, "n_samples": int(labels.numel()), "routers": {}}
    for router in ["static", "dynamic", "rule", "average"]:
        w = router_weights(probs, router)
        final = (w.unsqueeze(-1) * probs).sum(dim=1)
        result["routers"][router] = metrics(final, labels, w)
        print(f"    {router:<8} acc={result['routers'][router]['accuracy']:.4%} mean_w={result['routers'][router].get('mean_weights')}")
    return result


def maybe_subset(dataset, max_samples: int | None):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    data_root = PROJECT_ROOT / "data"
    tf = eval_transform()
    domains = []
    domains.append(("MNIST_test", datasets.MNIST(root=str(data_root), train=False, download=False, transform=tf), None))
    try:
        domains.append(("QMNIST_test10k", datasets.QMNIST(root=str(data_root), what="test10k", compat=True, download=False, transform=tf), None))
    except Exception as exc:
        print(f"skip QMNIST: {exc}")
    try:
        domains.append(("EMNIST_digits_test", datasets.EMNIST(root=str(data_root), split="digits", train=False, download=False, transform=eval_transform_emnist()), 10000))
    except Exception as exc:
        print(f"skip EMNIST: {exc}")
    try:
        domains.append(("USPS_train", datasets.USPS(root=str(data_root), train=True, download=False, transform=tf), None))
    except Exception as exc:
        print(f"skip USPS: {exc}")

    results = []
    for name, dataset, max_samples in domains:
        results.append(evaluate_domain(name, maybe_subset(dataset, max_samples), device))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
