"""Calibrate OOF predictions of a TestA specialist K-Fold experiment via class-prior shift.

The model over-predicts class 1 and under-predicts class 8 (see error analysis).
A logit shift  z'_c = log p_c - alpha * log(pred_count_c / true_count_c)
removes the systematic bias without retraining.

Usage:
    python scripts/calibrate_oof_priors.py --experiment-dir outputs_runs/<exp_id>
    python scripts/calibrate_oof_priors.py --experiment-dir outputs_runs/<exp_id> --alpha-grid 0.0 0.25 0.5 0.75 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_fold_probs(experiment_dir: Path, seed: int = 42, n_folds: int = 5):
    """Return list of (sample_ids, labels, probs) for each fold."""
    folds = []
    for fold_index in range(n_folds):
        path = experiment_dir / f"seed_{seed}" / f"fold_{fold_index}" / "predictions" / "validation_probabilities.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        folds.append({
            "sample_ids": list(payload["sample_ids"]),
            "labels": payload["labels"].to(torch.int64),
            "probs": payload["probabilities"].to(torch.float64),
            "fold_index": fold_index,
        })
    return folds


def compute_prior_from_folds(folds, n_classes: int = 10):
    """Compute (predicted_count / true_count) per class from a list of fold dicts."""
    pred_counts = np.zeros(n_classes, dtype=np.float64)
    true_counts = np.zeros(n_classes, dtype=np.float64)
    for f in folds:
        preds = f["probs"].argmax(dim=1).numpy()
        labels = f["labels"].numpy()
        for c in range(n_classes):
            pred_counts[c] += (preds == c).sum()
            true_counts[c] += (labels == c).sum()
    # Guard against zero divisions
    true_counts = np.maximum(true_counts, 1.0)
    ratio = pred_counts / true_counts
    return ratio, pred_counts, true_counts


def apply_logit_shift(probs: torch.Tensor, prior_ratio: np.ndarray, alpha: float) -> torch.Tensor:
    """Return softmax of (log probs - alpha * log prior_ratio)."""
    log_prior = torch.from_numpy(np.log(np.maximum(prior_ratio, 1e-9))).to(probs.dtype)
    logits = torch.log(probs.clamp_min(1e-12))
    shifted = logits - alpha * log_prior
    return torch.softmax(shifted, dim=1)


def evaluate(folds, alpha: float, prior_mode: str, n_classes: int = 10):
    """Return per-fold and overall accuracy after calibration with given alpha.

    prior_mode:
        - 'global': use the same prior estimated from all folds for every fold (mild leakage,
          but useful as an upper bound).
        - 'loo': for each fold, estimate prior from the other folds (no leakage).
    """
    n_folds = len(folds)
    all_labels = []
    all_preds_raw = []
    all_preds_cal = []
    fold_metrics = []

    if prior_mode == "global":
        global_ratio, _, _ = compute_prior_from_folds(folds, n_classes)
    elif prior_mode == "loo":
        global_ratio = None
    else:
        raise ValueError(f"unknown prior_mode: {prior_mode}")

    for i, f in enumerate(folds):
        if prior_mode == "global":
            ratio = global_ratio
        else:
            others = [g for j, g in enumerate(folds) if j != i]
            ratio, _, _ = compute_prior_from_folds(others, n_classes)
        calibrated = apply_logit_shift(f["probs"], ratio, alpha)
        preds_cal = calibrated.argmax(dim=1).numpy()
        preds_raw = f["probs"].argmax(dim=1).numpy()
        labels = f["labels"].numpy()
        fold_metrics.append({
            "fold_index": f["fold_index"],
            "n_samples": int(len(labels)),
            "acc_raw": float((preds_raw == labels).mean()),
            "acc_cal": float((preds_cal == labels).mean()),
            "prior_ratio": ratio.tolist(),
        })
        all_labels.append(labels)
        all_preds_raw.append(preds_raw)
        all_preds_cal.append(preds_cal)

    labels = np.concatenate(all_labels)
    preds_raw = np.concatenate(all_preds_raw)
    preds_cal = np.concatenate(all_preds_cal)

    per_class = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_class.append({
            "class": c,
            "n": int(mask.sum()),
            "acc_raw": float((preds_raw[mask] == c).mean()),
            "acc_cal": float((preds_cal[mask] == c).mean()),
            "pred_count_raw": int((preds_raw == c).sum()),
            "pred_count_cal": int((preds_cal == c).sum()),
        })

    return {
        "alpha": alpha,
        "prior_mode": prior_mode,
        "n_samples": int(len(labels)),
        "overall_acc_raw": float((preds_raw == labels).mean()),
        "overall_acc_cal": float((preds_cal == labels).mean()),
        "delta": float((preds_cal == labels).mean() - (preds_raw == labels).mean()),
        "per_fold": fold_metrics,
        "per_class": per_class,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--alpha-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--prior-mode", choices=["global", "loo", "both"], default="both")
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    exp_dir = args.experiment_dir
    if not exp_dir.is_absolute():
        exp_dir = (PROJECT_ROOT / exp_dir).resolve()
    folds = load_fold_probs(exp_dir, args.seed, args.n_folds)

    modes = ["global", "loo"] if args.prior_mode == "both" else [args.prior_mode]
    all_runs = []
    print(f"\n=== Experiment: {exp_dir.name} ===")
    print(f"Loaded {args.n_folds} folds, total samples = {sum(len(f['labels']) for f in folds)}")

    # Print raw class bias (just for context)
    ratio, pred_counts, true_counts = compute_prior_from_folds(folds)
    print("\nRaw predicted/true counts (all folds combined):")
    print(f"  {'class':>5} {'true':>5} {'pred':>5} {'ratio':>7}")
    for c in range(10):
        print(f"  {c:>5} {int(true_counts[c]):>5} {int(pred_counts[c]):>5} {ratio[c]:>7.3f}")

    print(f"\n{'mode':>7} {'alpha':>6} {'acc_raw':>8} {'acc_cal':>8} {'delta':>8}")
    for mode in modes:
        for alpha in args.alpha_grid:
            res = evaluate(folds, alpha, mode)
            all_runs.append(res)
            print(f"  {mode:>5} {alpha:>6.2f} {res['overall_acc_raw']:>8.4f} {res['overall_acc_cal']:>8.4f} {res['delta']:>+8.4f}")

    # Pick best
    best = max(all_runs, key=lambda r: r["overall_acc_cal"])
    print(f"\nBest: prior_mode={best['prior_mode']} alpha={best['alpha']:.2f} "
          f"acc {best['overall_acc_raw']:.4f} -> {best['overall_acc_cal']:.4f} (delta {best['delta']:+.4f})")
    print("\nPer-class change at best alpha:")
    print(f"  {'class':>5} {'n':>4} {'raw_acc':>8} {'cal_acc':>8} {'delta':>8} {'pred_raw':>9} {'pred_cal':>9}")
    for row in best["per_class"]:
        d = row["acc_cal"] - row["acc_raw"]
        print(f"  {row['class']:>5} {row['n']:>4} {row['acc_raw']:>8.4f} {row['acc_cal']:>8.4f} {d:>+8.4f} {row['pred_count_raw']:>9} {row['pred_count_cal']:>9}")

    if args.output_json is None:
        args.output_json = exp_dir / "calibration_summary.json"
    args.output_json.write_text(json.dumps({
        "experiment_dir": str(exp_dir),
        "raw_prior_ratio": ratio.tolist(),
        "raw_pred_counts": pred_counts.tolist(),
        "raw_true_counts": true_counts.tolist(),
        "runs": all_runs,
        "best": best,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
