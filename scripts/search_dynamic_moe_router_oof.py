"""Search lightweight dynamic MoE routing rules on TestA OOF probabilities.

The search is vectorized on GPU: OOF tensors are loaded once, routing features are
computed once, and parameter grids are evaluated in large tensor batches.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERTS = [
    {
        "label": "wide_resnet_tiny_raw_seed42",
        "weight": 0.7,
        "oof": PROJECT_ROOT / "outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_anti1_seed2026",
        "weight": 0.2,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_raw_seed3407",
        "weight": 0.1,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_raw_seed3407_e60/oof/oof_probabilities.pt",
    },
]

PARAM_GRID = {
    "confidence": [-2.0, -1.0, 0.0, 0.75, 1.5, 2.5, 3.5, 5.0],
    "margin": [-1.0, 0.0, 0.75, 1.5, 2.5, 3.5],
    "disagreement": [-1.0, 0.0, 0.5, 1.0, 1.75, 2.5, 3.5],
    "anti1_boost": [0.0, 0.25, 0.5, 1.0, 1.5],
}


def load_oof(device: str):
    probabilities = []
    labels = None
    sample_ids = None
    for expert in EXPERTS:
        payload = torch.load(expert["oof"], map_location="cpu")
        if labels is None:
            labels = payload["labels"].long()
            sample_ids = payload["sample_ids"]
        elif sample_ids != payload["sample_ids"] or not torch.equal(labels, payload["labels"].long()):
            raise ValueError(f"OOF sample order mismatch: {expert['label']}")
        probabilities.append(payload["probabilities"].float())
    return torch.stack(probabilities, dim=1).to(device), labels.to(device), labels.cpu().numpy()


def compute_features(probs: torch.Tensor, base_weights: torch.Tensor):
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]
    mean_distribution = torch.sum(probs * base_weights.view(1, -1, 1), dim=1, keepdim=True).clamp_min(1e-9)
    disagreement = torch.sum(probs.clamp_min(1e-9) * torch.log(probs.clamp_min(1e-9) / mean_distribution), dim=-1)

    confidence = confidence - confidence.mean(dim=1, keepdim=True)
    margin = margin - margin.mean(dim=1, keepdim=True)
    disagreement = disagreement - disagreement.mean(dim=1, keepdim=True)

    fixed_probs = torch.sum(probs * base_weights.view(1, -1, 1), dim=1)
    fixed_pred = fixed_probs.argmax(dim=1)
    anti1_signal = (fixed_pred == 1).float().view(-1, 1) * torch.tensor(
        [[0.0, 1.0, 0.0]], device=probs.device, dtype=probs.dtype
    )
    return confidence, margin, disagreement, anti1_signal


def diagnostics(predictions: torch.Tensor, labels: torch.Tensor):
    accuracy = float((predictions == labels).float().mean().item())
    result = {"accuracy": accuracy, "n_samples": int(labels.numel())}
    true_counts = torch.bincount(labels, minlength=10).float()
    pred_counts = torch.bincount(predictions, minlength=10).float()
    correct_by_class = torch.bincount(labels[predictions == labels], minlength=10).float()
    per_class_acc = correct_by_class / true_counts.clamp_min(1)
    result["class_1_overprediction_ratio"] = float((pred_counts[1] / true_counts[1].clamp_min(1)).item())
    result["class_8_prediction_ratio"] = float((pred_counts[8] / true_counts[8].clamp_min(1)).item())
    result["class_1_accuracy"] = float(per_class_acc[1].item())
    result["class_8_accuracy"] = float(per_class_acc[8].item())
    result["total_x_to_1_errors"] = int(((predictions == 1) & (labels != 1)).sum().item())
    return result


def make_param_tensor(device: str):
    rows = list(product(PARAM_GRID["confidence"], PARAM_GRID["margin"], PARAM_GRID["disagreement"], PARAM_GRID["anti1_boost"]))
    return torch.tensor(rows, device=device, dtype=torch.float32), rows


def predict_for_params(
    probs: torch.Tensor,
    base_weights: torch.Tensor,
    features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    params: torch.Tensor,
    batch_size: int = 256,
):
    confidence, margin, disagreement, anti1_signal = features
    log_base = torch.log(base_weights).view(1, 1, -1)
    all_predictions = []
    all_weights = []
    for start in range(0, params.shape[0], batch_size):
        chunk = params[start : start + batch_size]
        score = (
            log_base
            + chunk[:, 0].view(-1, 1, 1) * confidence.unsqueeze(0)
            + chunk[:, 1].view(-1, 1, 1) * margin.unsqueeze(0)
            - chunk[:, 2].view(-1, 1, 1) * disagreement.unsqueeze(0)
            + chunk[:, 3].view(-1, 1, 1) * anti1_signal.unsqueeze(0)
        )
        dynamic_weights = torch.softmax(score, dim=-1)
        ensemble_probs = torch.einsum("gne,nec->gnc", dynamic_weights, probs)
        all_predictions.append(ensemble_probs.argmax(dim=-1))
        all_weights.append(dynamic_weights.mean(dim=1))
    return torch.cat(all_predictions, dim=0), torch.cat(all_weights, dim=0)


def evaluate_grid(probs, labels, features, base_weights, params, indices=None):
    if indices is not None:
        probs = probs[indices]
        labels = labels[indices]
        features = tuple(feature[indices] for feature in features)
    predictions, mean_weights = predict_for_params(probs, base_weights, features, params)
    accuracies = (predictions == labels.view(1, -1)).float().mean(dim=1)
    return accuracies, predictions, mean_weights


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probs, labels, labels_np = load_oof(device)
    base_weights = torch.tensor([expert["weight"] for expert in EXPERTS], device=device, dtype=torch.float32)
    features = compute_features(probs, base_weights)
    params, param_rows = make_param_tensor(device)

    fixed_predictions = torch.sum(probs * base_weights.view(1, -1, 1), dim=1).argmax(dim=1)
    fixed_diag = diagnostics(fixed_predictions, labels)

    accuracies, predictions, mean_weights = evaluate_grid(probs, labels, features, base_weights, params)
    top_values, top_indices = torch.topk(accuracies, k=min(20, len(param_rows)))
    top_candidates = []
    for rank, (value, index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
        candidate_predictions = predictions[index]
        diag = diagnostics(candidate_predictions, labels)
        top_candidates.append(
            {
                "rank": rank,
                "params": dict(zip(PARAM_GRID.keys(), param_rows[index])),
                "mean_weights": mean_weights[index].detach().cpu().tolist(),
                "diagnostics": diag,
            }
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    val_predictions = torch.empty_like(labels)
    for fold, (train_idx_np, val_idx_np) in enumerate(skf.split(np.zeros_like(labels_np), labels_np)):
        train_idx = torch.tensor(train_idx_np, device=device, dtype=torch.long)
        val_idx = torch.tensor(val_idx_np, device=device, dtype=torch.long)
        train_acc, _, _ = evaluate_grid(probs, labels, features, base_weights, params, train_idx)
        best_index = int(torch.argmax(train_acc).item())
        val_acc, val_preds, val_mean_weights = evaluate_grid(probs, labels, features, base_weights, params[best_index : best_index + 1], val_idx)
        val_predictions[val_idx] = val_preds[0]
        cv_rows.append(
            {
                "fold": fold,
                "train_accuracy": float(train_acc[best_index].item()),
                "val_accuracy": float(val_acc[0].item()),
                "params": dict(zip(PARAM_GRID.keys(), param_rows[best_index])),
                "mean_weights_on_val": val_mean_weights[0].detach().cpu().tolist(),
            }
        )

    cv_diag = diagnostics(val_predictions, labels)

    output = {
        "device": device,
        "experts": [{"label": e["label"], "base_weight": e["weight"]} for e in EXPERTS],
        "n_params": len(param_rows),
        "fixed_baseline": fixed_diag,
        "best_in_sample": top_candidates[0],
        "top_candidates": top_candidates,
        "router_cv": {"diagnostics": cv_diag, "folds": cv_rows},
    }

    output_dir = PROJECT_ROOT / "outputs_runs/dynamic_moe_router"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "search_summary.json"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_dir / "top_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "accuracy",
                "class_1_overprediction_ratio",
                "class_8_accuracy",
                "total_x_to_1_errors",
                "confidence",
                "margin",
                "disagreement",
                "anti1_boost",
                "mean_weight_wide",
                "mean_weight_anti1",
                "mean_weight_raw",
            ],
        )
        writer.writeheader()
        for candidate in top_candidates:
            params_dict = candidate["params"]
            diag = candidate["diagnostics"]
            weights = candidate["mean_weights"]
            writer.writerow(
                {
                    "rank": candidate["rank"],
                    "accuracy": diag["accuracy"],
                    "class_1_overprediction_ratio": diag["class_1_overprediction_ratio"],
                    "class_8_accuracy": diag["class_8_accuracy"],
                    "total_x_to_1_errors": diag["total_x_to_1_errors"],
                    "confidence": params_dict["confidence"],
                    "margin": params_dict["margin"],
                    "disagreement": params_dict["disagreement"],
                    "anti1_boost": params_dict["anti1_boost"],
                    "mean_weight_wide": weights[0],
                    "mean_weight_anti1": weights[1],
                    "mean_weight_raw": weights[2],
                }
            )

    print(json.dumps({"fixed_baseline": fixed_diag, "best_in_sample": top_candidates[0], "router_cv": output["router_cv"]}, indent=2, ensure_ascii=False))
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
