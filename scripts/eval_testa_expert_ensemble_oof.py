"""Soft-vote ensemble across multiple TestA specialist OOF predictions.

Reads each experiment's outputs_runs/<exp>/oof/oof_probabilities.pt produced by
scripts/eval_testa_specialist_oof.py, verifies that sample_ids and labels match
exactly across experts, then explores either an equal-weight or a grid-search
weighted combination.

The fundamental diagnostic question this script answers is NOT just "what is the
ensemble accuracy", but specifically:

  * does the ensemble reduce the class-1 over-prediction bias?
  * does class-8 recover toward parity?
  * how many of the systematic X -> 1 errors actually go away?

These three signals are reported alongside overall accuracy, top-k confusion,
and per-class accuracy so that a small overall accuracy gain that is purely
random variance can be told apart from a genuine bias correction.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ensemble_predict import average_probabilities
from src.error_analysis import per_class_accuracy, top_confused_pairs


SYSTEMATIC_TARGETS_TO_ONE = [8, 9, 6, 5, 3, 2, 4]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[],
        help="experiment_id list; resolved to outputs_runs/<id>/oof/oof_probabilities.pt",
    )
    parser.add_argument(
        "--oof-paths",
        type=Path,
        nargs="+",
        default=[],
        help="explicit OOF probabilities paths; mutually exclusive with --experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs_runs" / "expert_ensemble_analysis",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="enumerate Dirichlet-grid weights summing to 1 with given step",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="grid step for --grid-search (e.g. 0.1 or 0.05)",
    )
    parser.add_argument(
        "--equal-weight-only",
        action="store_true",
        help="only evaluate uniform weights (overrides --grid-search)",
    )
    return parser.parse_args()


def resolve_oof_paths(args) -> list[tuple[str, Path]]:
    """Return list of (label, path) for each requested expert."""
    if args.experiments and args.oof_paths:
        raise SystemExit("Provide either --experiments or --oof-paths, not both.")
    if not args.experiments and not args.oof_paths:
        raise SystemExit("Must provide --experiments or --oof-paths.")

    project_root = args.project_root.resolve()
    pairs: list[tuple[str, Path]] = []
    if args.experiments:
        for exp_id in args.experiments:
            oof_path = project_root / "outputs_runs" / exp_id / "oof" / "oof_probabilities.pt"
            if not oof_path.exists():
                config_yaml = f"experiments/{exp_id}.yaml"
                raise FileNotFoundError(
                    f"未找到 OOF 概率文件：{oof_path}\n"
                    f"请先运行：\n"
                    f"  python scripts/eval_testa_specialist_oof.py --config {config_yaml}"
                )
            pairs.append((exp_id, oof_path))
    else:
        for path in args.oof_paths:
            path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(f"未找到 OOF 概率文件：{path}")
            pairs.append((path.parent.parent.name, path))
    return pairs


def load_expert(label: str, oof_path: Path) -> dict:
    payload = torch.load(oof_path, map_location="cpu", weights_only=False)
    sample_ids = [int(sid) for sid in payload["sample_ids"]]
    labels = payload["labels"].cpu().long()
    probabilities = payload["probabilities"].cpu().float()
    if probabilities.ndim != 2 or probabilities.shape[1] != 10:
        raise ValueError(f"{label}: probabilities shape={tuple(probabilities.shape)} 不是 [N,10]")
    if labels.numel() != probabilities.shape[0]:
        raise ValueError(f"{label}: labels 长度 {labels.numel()} 与 probabilities 行数 {probabilities.shape[0]} 不一致")
    if len(sample_ids) != probabilities.shape[0]:
        raise ValueError(f"{label}: sample_ids 长度 {len(sample_ids)} 与 probabilities 行数 {probabilities.shape[0]} 不一致")
    return {"label": label, "path": oof_path, "sample_ids": sample_ids, "labels": labels, "probabilities": probabilities}


def align_and_check(experts: list[dict]) -> tuple[list[int], torch.Tensor, list[torch.Tensor]]:
    """Sort by sample_id, then verify every expert agrees on ids/labels.

    Returns canonical sorted (sample_ids, labels, [probabilities per expert]).
    """
    if not experts:
        raise ValueError("没有任何专家可加载")

    canonical_ids = None
    canonical_labels = None
    aligned_probs: list[torch.Tensor] = []
    for expert in experts:
        order = sorted(range(len(expert["sample_ids"])), key=lambda i: expert["sample_ids"][i])
        ids_sorted = [expert["sample_ids"][i] for i in order]
        labels_sorted = expert["labels"][order]
        probs_sorted = expert["probabilities"][order]
        if canonical_ids is None:
            canonical_ids = ids_sorted
            canonical_labels = labels_sorted
        else:
            if len(ids_sorted) != len(canonical_ids):
                raise RuntimeError(
                    f"专家 {expert['label']} 样本数 {len(ids_sorted)} 与基准 {len(canonical_ids)} 不一致"
                )
            for i, (a, b) in enumerate(zip(ids_sorted, canonical_ids)):
                if a != b:
                    raise RuntimeError(
                        f"专家 {expert['label']} 在排序后第 {i} 个 sample_id 不一致 (got {a}, expected {b})"
                    )
            if not torch.equal(labels_sorted, canonical_labels):
                first_bad = int(((labels_sorted != canonical_labels).nonzero(as_tuple=True)[0])[0].item())
                raise RuntimeError(
                    f"专家 {expert['label']} 在 sample_id={canonical_ids[first_bad]} 上的标签 "
                    f"{int(labels_sorted[first_bad].item())} 与基准 {int(canonical_labels[first_bad].item())} 不一致"
                )
        aligned_probs.append(probs_sorted)
    return canonical_ids, canonical_labels, aligned_probs


def enumerate_grid_weights(k: int, step: float) -> list[tuple[float, ...]]:
    """Enumerate non-negative weight vectors of length k summing exactly to 1 on a step lattice."""
    if step <= 0 or step > 1:
        raise ValueError(f"grid_step must be in (0,1], got {step}")
    n_units = int(round(1.0 / step))
    if abs(n_units * step - 1.0) > 1e-6:
        raise ValueError(f"grid_step={step} doesn't divide 1.0 evenly")
    combos: list[tuple[float, ...]] = []
    if k == 1:
        return [(1.0,)]

    def recurse(prefix: list[int], remaining_slots: int, remaining_units: int):
        if remaining_slots == 1:
            prefix_extended = prefix + [remaining_units]
            combos.append(tuple(round(v * step, 6) for v in prefix_extended))
            return
        for chosen in range(remaining_units + 1):
            recurse(prefix + [chosen], remaining_slots - 1, remaining_units - chosen)

    recurse([], k, n_units)
    return combos


def compute_diagnostics(labels: torch.Tensor, probabilities: torch.Tensor, num_classes: int = 10) -> dict:
    predictions = probabilities.argmax(dim=1)
    matches = (predictions == labels).float()
    overall_acc = float(matches.mean().item()) if labels.numel() else 0.0

    true_counts = [int((labels == c).sum().item()) for c in range(num_classes)]
    pred_counts = [int((predictions == c).sum().item()) for c in range(num_classes)]
    ratios = [
        (pred_counts[c] / true_counts[c]) if true_counts[c] else None
        for c in range(num_classes)
    ]

    per_class = per_class_accuracy(labels, predictions, num_classes=num_classes)
    confused = top_confused_pairs(labels, predictions, num_classes=num_classes, top_k=15)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
        cm[int(t), int(p)] += 1

    total_x_to_1 = int(sum(cm[t, 1] for t in range(num_classes) if t != 1))
    pair_errors_to_1 = {f"{t}->1": int(cm[t, 1]) for t in SYSTEMATIC_TARGETS_TO_ONE}
    class_8_accuracy = next((row["accuracy"] for row in per_class if row["class"] == 8), None)
    class_1_accuracy = next((row["accuracy"] for row in per_class if row["class"] == 1), None)

    return {
        "overall_accuracy": overall_acc,
        "n_samples": int(labels.numel()),
        "true_count_by_class": true_counts,
        "predicted_count_by_class": pred_counts,
        "predicted_true_ratio_by_class": ratios,
        "class_1_overprediction_ratio": ratios[1],
        "class_8_underprediction_ratio": ratios[8],
        "class_1_accuracy": class_1_accuracy,
        "class_8_accuracy": class_8_accuracy,
        "total_x_to_1_errors": total_x_to_1,
        "pair_errors_to_1": pair_errors_to_1,
        "per_class_accuracy": per_class,
        "top_confused_pairs": confused,
        "confusion_matrix": cm.tolist(),
    }


def write_csv_summary(rows: list[dict], path: Path, expert_labels: list[str]) -> None:
    fieldnames = (
        [f"w_{label}" for label in expert_labels]
        + [
            "overall_accuracy",
            "class_1_overprediction_ratio",
            "class_8_underprediction_ratio",
            "class_1_accuracy",
            "class_8_accuracy",
            "total_x_to_1_errors",
        ]
        + [f"err_{t}_to_1" for t in SYSTEMATIC_TARGETS_TO_ONE]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {f"w_{lab}": w for lab, w in zip(expert_labels, row["weights"])}
            diag = row["diagnostics"]
            flat["overall_accuracy"] = diag["overall_accuracy"]
            flat["class_1_overprediction_ratio"] = diag["class_1_overprediction_ratio"]
            flat["class_8_underprediction_ratio"] = diag["class_8_underprediction_ratio"]
            flat["class_1_accuracy"] = diag["class_1_accuracy"]
            flat["class_8_accuracy"] = diag["class_8_accuracy"]
            flat["total_x_to_1_errors"] = diag["total_x_to_1_errors"]
            for t in SYSTEMATIC_TARGETS_TO_ONE:
                flat[f"err_{t}_to_1"] = diag["pair_errors_to_1"][f"{t}->1"]
            writer.writerow(flat)


def write_predictions_csv(path: Path, sample_ids: list[int], labels: torch.Tensor, probabilities: torch.Tensor) -> None:
    predictions = probabilities.argmax(dim=1)
    confidences = probabilities.max(dim=1).values
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "label", "prediction", "confidence", "correct"] + [f"prob_{c}" for c in range(10)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids):
            writer.writerow({
                "sample_id": int(sample_id),
                "label": int(labels[index].item()),
                "prediction": int(predictions[index].item()),
                "confidence": float(confidences[index].item()),
                "correct": bool(int(predictions[index].item()) == int(labels[index].item())),
                **{f"prob_{c}": float(probabilities[index, c].item()) for c in range(10)},
            })


def write_confusion_csv(path: Path, cm: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + [f"pred_{c}" for c in range(10)])
        for true_label, row in enumerate(cm):
            writer.writerow([true_label] + list(row))


def select_best(rows: list[dict], criterion: str) -> dict:
    if criterion == "overall_accuracy":
        return max(rows, key=lambda r: r["diagnostics"]["overall_accuracy"])
    if criterion == "class_8_accuracy":
        return max(rows, key=lambda r: (r["diagnostics"]["class_8_accuracy"] or -1.0))
    if criterion == "class_1_overprediction_ratio_min":
        def score(r):
            ratio = r["diagnostics"]["class_1_overprediction_ratio"]
            return abs((ratio or 1.0) - 1.0)
        return min(rows, key=score)
    if criterion == "total_x_to_1_min":
        return min(rows, key=lambda r: r["diagnostics"]["total_x_to_1_errors"])
    raise ValueError(criterion)


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if "outputs_submission" in str(output_dir):
        raise SystemExit("禁止写入 outputs_submission 路径。请改 --output-dir。")
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = resolve_oof_paths(args)
    experts = [load_expert(label, path) for label, path in pairs]
    sample_ids, labels, aligned_probs = align_and_check(experts)
    expert_labels = [e["label"] for e in experts]
    k = len(experts)

    # Build weight grid
    if args.equal_weight_only or k == 1:
        weight_grid: list[tuple[float, ...]] = [(1.0 / k,) * k]
    elif args.grid_search:
        weight_grid = enumerate_grid_weights(k, args.grid_step)
    else:
        weight_grid = [(1.0 / k,) * k]

    rows: list[dict] = []
    for weights in weight_grid:
        weight_tensor = torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1)
        stacked = torch.stack(aligned_probs, dim=0)  # [K, N, 10]
        fused = (weight_tensor * stacked).sum(dim=0)
        diag = compute_diagnostics(labels, fused)
        rows.append({
            "weights": list(weights),
            "diagnostics": diag,
        })

    best_overall = select_best(rows, "overall_accuracy")
    best_min_class1 = select_best(rows, "class_1_overprediction_ratio_min")
    best_max_class8 = select_best(rows, "class_8_accuracy")
    best_min_x_to_1 = select_best(rows, "total_x_to_1_min")

    # Per-expert diagnostics (uniform weight is just a sanity reference; we report each alone)
    per_expert_diagnostics = []
    for index, expert in enumerate(experts):
        diag = compute_diagnostics(expert["labels"], expert["probabilities"])
        per_expert_diagnostics.append({
            "expert": expert["label"],
            "oof_path": str(expert["path"]),
            "diagnostics": diag,
        })

    # Best ensemble probabilities + predictions
    best_weights = best_overall["weights"]
    weight_tensor = torch.tensor(best_weights, dtype=torch.float32).view(-1, 1, 1)
    stacked = torch.stack(aligned_probs, dim=0)
    best_probs = (weight_tensor * stacked).sum(dim=0)

    summary = {
        "experts": [
            {"label": e["label"], "oof_path": str(e["path"])} for e in experts
        ],
        "n_experts": k,
        "n_samples": int(labels.numel()),
        "grid_step": args.grid_step if args.grid_search and not args.equal_weight_only else None,
        "n_weight_candidates": len(rows),
        "per_expert_diagnostics": per_expert_diagnostics,
        "candidates": rows,
        "best_overall_accuracy": best_overall,
        "best_min_class_1_overprediction": best_min_class1,
        "best_max_class_8_accuracy": best_max_class8,
        "best_min_total_x_to_1": best_min_x_to_1,
    }

    summary_path = output_dir / "expert_ensemble_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "expert_ensemble_summary.csv"
    write_csv_summary(rows, csv_path, expert_labels)

    write_predictions_csv(
        output_dir / "best_ensemble_oof_predictions.csv",
        sample_ids,
        labels,
        best_probs,
    )

    torch.save(
        {
            "sample_ids": sample_ids,
            "labels": labels,
            "probabilities": best_probs,
            "weights": best_weights,
            "experts": expert_labels,
        },
        output_dir / "best_ensemble_oof_probabilities.pt",
    )

    write_confusion_csv(output_dir / "confusion_matrix.csv", best_overall["diagnostics"]["confusion_matrix"])

    per_expert_csv = output_dir / "per_expert_diagnostics.csv"
    with per_expert_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "expert",
            "overall_accuracy",
            "class_1_overprediction_ratio",
            "class_8_underprediction_ratio",
            "class_1_accuracy",
            "class_8_accuracy",
            "total_x_to_1_errors",
        ] + [f"err_{t}_to_1" for t in SYSTEMATIC_TARGETS_TO_ONE]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in per_expert_diagnostics:
            diag = entry["diagnostics"]
            writer.writerow({
                "expert": entry["expert"],
                "overall_accuracy": diag["overall_accuracy"],
                "class_1_overprediction_ratio": diag["class_1_overprediction_ratio"],
                "class_8_underprediction_ratio": diag["class_8_underprediction_ratio"],
                "class_1_accuracy": diag["class_1_accuracy"],
                "class_8_accuracy": diag["class_8_accuracy"],
                "total_x_to_1_errors": diag["total_x_to_1_errors"],
                **{f"err_{t}_to_1": diag["pair_errors_to_1"][f"{t}->1"] for t in SYSTEMATIC_TARGETS_TO_ONE},
            })

    print(json.dumps({
        "n_experts": k,
        "n_candidates": len(rows),
        "best_overall_accuracy": best_overall["diagnostics"]["overall_accuracy"],
        "best_overall_weights": best_overall["weights"],
        "best_overall_class_1_ratio": best_overall["diagnostics"]["class_1_overprediction_ratio"],
        "best_overall_class_8_accuracy": best_overall["diagnostics"]["class_8_accuracy"],
        "best_overall_total_x_to_1": best_overall["diagnostics"]["total_x_to_1_errors"],
        "output_dir": str(output_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
