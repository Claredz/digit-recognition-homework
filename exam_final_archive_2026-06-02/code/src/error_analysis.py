from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def per_class_accuracy(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int = 10) -> list[dict]:
    rows = []
    for label in range(num_classes):
        mask = labels == label
        total = int(mask.sum().item())
        correct = int((predictions[mask] == labels[mask]).sum().item()) if total else 0
        rows.append({
            "class": label,
            "correct": correct,
            "total": total,
            "accuracy": (correct / total) if total else None,
        })
    return rows


def top_confused_pairs(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int = 10, top_k: int = 10) -> list[dict]:
    matrix = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=list(range(num_classes)))
    rows = []
    for true_label in range(num_classes):
        for predicted_label in range(num_classes):
            if true_label == predicted_label:
                continue
            count = int(matrix[true_label, predicted_label])
            if count:
                rows.append({"true": true_label, "predicted": predicted_label, "count": count})
    return sorted(rows, key=lambda row: row["count"], reverse=True)[:top_k]


def low_confidence_samples(sample_ids: list[int | str], probabilities: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor | None = None, top_k: int = 50) -> list[dict]:
    confidences = probabilities.max(dim=1).values.cpu()
    order = torch.argsort(confidences)[: min(top_k, len(confidences))]
    rows = []
    for index in order.tolist():
        row = {
            "sample_id": sample_ids[index],
            "prediction": int(predictions[index].item()),
            "confidence": float(confidences[index].item()),
        }
        if labels is not None:
            row["label"] = int(labels[index].item())
            row["correct"] = bool(predictions[index].item() == labels[index].item())
        rows.append(row)
    return rows


def high_confidence_wrong_samples(sample_ids: list[int | str], probabilities: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor, top_k: int = 50) -> list[dict]:
    confidences = probabilities.max(dim=1).values.cpu()
    wrong = (predictions.cpu() != labels.cpu()).nonzero(as_tuple=False).flatten()
    if wrong.numel() == 0:
        return []
    wrong_confidences = confidences[wrong]
    order = torch.argsort(wrong_confidences, descending=True)[: min(top_k, wrong.numel())]
    rows = []
    for local_index in order.tolist():
        index = int(wrong[local_index].item())
        rows.append({
            "sample_id": sample_ids[index],
            "label": int(labels[index].item()),
            "prediction": int(predictions[index].item()),
            "confidence": float(confidences[index].item()),
        })
    return rows


def raw_preprocess_disagreements(sample_ids: list[int | str], raw_predictions: torch.Tensor, preprocess_predictions: torch.Tensor, labels: torch.Tensor | None = None) -> list[dict]:
    disagreements = (raw_predictions.cpu() != preprocess_predictions.cpu()).nonzero(as_tuple=False).flatten()
    rows = []
    for index in disagreements.tolist():
        row = {
            "sample_id": sample_ids[index],
            "raw_prediction": int(raw_predictions[index].item()),
            "preprocess_prediction": int(preprocess_predictions[index].item()),
        }
        if labels is not None:
            row["label"] = int(labels[index].item())
        rows.append(row)
    return rows


def save_per_class_error_grids(images: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor, output_dir: Path, max_per_class: int = 8) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for label in range(10):
        indices = ((labels == label) & (predictions != labels)).nonzero(as_tuple=False).flatten().tolist()
        if not indices:
            continue
        chosen = indices[:max_per_class]
        cols = min(4, len(chosen))
        rows = int(np.ceil(len(chosen) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes_array = np.array(axes).reshape(-1)
        for axis, sample_index in zip(axes_array, chosen):
            image = (images[sample_index] * 0.5 + 0.5).clamp(0, 1).squeeze(0).cpu().numpy()
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"真:{int(labels[sample_index])} 预测:{int(predictions[sample_index])}")
            axis.axis("off")
        for axis in axes_array[len(chosen):]:
            axis.axis("off")
        fig.tight_layout()
        path = output_dir / f"class_{label}_errors.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(str(path))
    return saved


def save_error_analysis_bundle(
    output_dir: Path,
    sample_ids: list[int | str],
    images: torch.Tensor,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    preprocess_probabilities: torch.Tensor | None = None,
    num_classes: int = 10,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(dim=1).cpu()
    labels = labels.cpu()
    per_class = per_class_accuracy(labels, predictions, num_classes=num_classes)
    confused = top_confused_pairs(labels, predictions, num_classes=num_classes)
    low_conf = low_confidence_samples(sample_ids, probabilities.cpu(), predictions, labels)
    high_wrong = high_confidence_wrong_samples(sample_ids, probabilities.cpu(), predictions, labels)
    disagreements = []
    if preprocess_probabilities is not None:
        disagreements = raw_preprocess_disagreements(
            sample_ids,
            predictions,
            preprocess_probabilities.argmax(dim=1).cpu(),
            labels,
        )
    grid_paths = save_per_class_error_grids(images.cpu(), labels, predictions, output_dir / "class_error_grids")
    payload = {
        "per_class_accuracy": per_class,
        "top_confused_pairs": confused,
        "low_confidence_samples": low_conf,
        "high_confidence_wrong_samples": high_wrong,
        "raw_preprocess_disagreements": disagreements,
        "class_error_grids": grid_paths,
    }
    (output_dir / "error_analysis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "per_class_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(per_class)
    return payload


# ---------------------------------------------------------------------------
# Phase A.2 — extended diagnostics for the multi-expert TestA system.
#
# The baseline OOF (testa_partial_init_lr1e4_mixup01_erasing005_e40, OOF ≈ 0.7420)
# exhibits a SYSTEMATIC class-1 over-prediction bias: predicted/true ratio ≈ 1.27,
# with ~165 X→1 errors concentrated on {8,9,6,5,3,2,4}. These functions surface
# that bias as first-class metrics so we can:
#
#   1. tell a real bias correction apart from a random accuracy fluctuation;
#   2. quantify error overlap between candidate experts (the ensemble payoff is
#      large only when their mistakes are different);
#   3. detect when a "better OOF" config is just memorising harder samples with
#      high but wrong confidence.
#
# Every function below operates on CPU tensors / lists; callers are responsible
# for moving from GPU. Sample_ids are propagated wherever provided.
# ---------------------------------------------------------------------------

SYSTEMATIC_TARGETS_TO_ONE: list[int] = [8, 9, 6, 5, 3, 2, 4]


def _ensure_tensor_long(values) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().long()
    return torch.as_tensor(values, dtype=torch.long)


def class_overprediction_ratio(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    target_class: int = 1,
) -> float | None:
    """predicted_count / true_count for a given class.

    Returns None when there are zero true samples (avoids /0).
    """
    predictions = _ensure_tensor_long(predictions)
    labels = _ensure_tensor_long(labels)
    true_count = int((labels == target_class).sum().item())
    if true_count == 0:
        return None
    pred_count = int((predictions == target_class).sum().item())
    return pred_count / true_count


def class_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
) -> float | None:
    """Per-class recall (a.k.a. class accuracy)."""
    predictions = _ensure_tensor_long(predictions)
    labels = _ensure_tensor_long(labels)
    mask = labels == target_class
    total = int(mask.sum().item())
    if total == 0:
        return None
    correct = int((predictions[mask] == labels[mask]).sum().item())
    return correct / total


def x_to_target_errors(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    target_class: int = 1,
    num_classes: int = 10,
) -> dict:
    """Count how many samples of each non-target class are misclassified as target_class.

    Returns:
        {
            "target_class": int,
            "total_x_to_target": int,                 # sum across all non-target classes
            "pair_errors": {"8->1": n, "9->1": n, ...},  # full enumeration
            "systematic_pair_errors": {"8->1": n, ...},  # restricted to SYSTEMATIC_TARGETS_TO_ONE
        }
    """
    predictions = _ensure_tensor_long(predictions)
    labels = _ensure_tensor_long(labels)
    pair_errors: dict[str, int] = {}
    total = 0
    for source_class in range(num_classes):
        if source_class == target_class:
            continue
        mask = (labels == source_class) & (predictions == target_class)
        count = int(mask.sum().item())
        pair_errors[f"{source_class}->{target_class}"] = count
        total += count
    systematic = {
        f"{cls}->{target_class}": pair_errors.get(f"{cls}->{target_class}", 0)
        for cls in SYSTEMATIC_TARGETS_TO_ONE
        if cls != target_class
    }
    return {
        "target_class": target_class,
        "total_x_to_target": total,
        "pair_errors": pair_errors,
        "systematic_pair_errors": systematic,
    }


def error_overlap(
    sample_ids_a: list[int | str],
    predictions_a: torch.Tensor,
    labels_a: torch.Tensor,
    sample_ids_b: list[int | str],
    predictions_b: torch.Tensor,
    labels_b: torch.Tensor,
    target_class: int = 1,
) -> dict:
    """Compute how much two predictors agree on their MISTAKES (not predictions).

    Sample alignment: we intersect by sample_id; if alignment fails (e.g. one set
    is a strict superset), we still report on the intersection but flag a warning.

    Returned dict:
        {
            "n_intersect": int,
            "n_errors_a": int,
            "n_errors_b": int,
            "n_shared_errors": int,
            "shared_error_ratio_over_a": float,   # n_shared_errors / n_errors_a
            "shared_error_ratio_over_b": float,
            "n_x_to_target_a": int,
            "n_x_to_target_b": int,
            "n_shared_x_to_target": int,
            "shared_x_to_target_ratio_over_a": float,
            "shared_x_to_target_ratio_over_b": float,
            "ids_unmatched_a": int,               # in a but not b
            "ids_unmatched_b": int,
        }

    Lower shared_error_ratio means MORE complementary errors → ensemble more
    likely to pay off.
    """
    predictions_a = _ensure_tensor_long(predictions_a)
    labels_a = _ensure_tensor_long(labels_a)
    predictions_b = _ensure_tensor_long(predictions_b)
    labels_b = _ensure_tensor_long(labels_b)

    map_a = {int(sid): index for index, sid in enumerate(sample_ids_a)}
    map_b = {int(sid): index for index, sid in enumerate(sample_ids_b)}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    only_a = set(map_a.keys()) - set(map_b.keys())
    only_b = set(map_b.keys()) - set(map_a.keys())

    errors_a: set[int] = set()
    errors_b: set[int] = set()
    x_to_target_a: set[int] = set()
    x_to_target_b: set[int] = set()
    for sid in common:
        idx_a = map_a[sid]
        idx_b = map_b[sid]
        if int(labels_a[idx_a]) != int(labels_b[idx_b]):
            continue  # treat label-mismatch as not directly comparable
        true_label = int(labels_a[idx_a])
        pred_a = int(predictions_a[idx_a])
        pred_b = int(predictions_b[idx_b])
        if pred_a != true_label:
            errors_a.add(sid)
        if pred_b != true_label:
            errors_b.add(sid)
        if pred_a == target_class and true_label != target_class:
            x_to_target_a.add(sid)
        if pred_b == target_class and true_label != target_class:
            x_to_target_b.add(sid)

    shared_errors = errors_a & errors_b
    shared_x = x_to_target_a & x_to_target_b

    def _safe_ratio(num: int, denom: int) -> float | None:
        if denom == 0:
            return None
        return num / denom

    return {
        "n_intersect": len(common),
        "n_errors_a": len(errors_a),
        "n_errors_b": len(errors_b),
        "n_shared_errors": len(shared_errors),
        "shared_error_ratio_over_a": _safe_ratio(len(shared_errors), len(errors_a)),
        "shared_error_ratio_over_b": _safe_ratio(len(shared_errors), len(errors_b)),
        "n_x_to_target_a": len(x_to_target_a),
        "n_x_to_target_b": len(x_to_target_b),
        "n_shared_x_to_target": len(shared_x),
        "shared_x_to_target_ratio_over_a": _safe_ratio(len(shared_x), len(x_to_target_a)),
        "shared_x_to_target_ratio_over_b": _safe_ratio(len(shared_x), len(x_to_target_b)),
        "ids_unmatched_a": len(only_a),
        "ids_unmatched_b": len(only_b),
        "target_class": target_class,
    }


def confidence_distribution_wrong(
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 10,
) -> dict:
    """Histogram of max-prob (confidence) restricted to wrong predictions.

    A high-confidence-wrong tail is the symptom of a memorising, NOT generalising,
    expert; we want this distribution to look skewed toward LOW confidence, not high.

    Returns:
        {
            "bin_edges": [0.0, 0.1, ..., 1.0],
            "counts": [n_in_bin_0, ..., n_in_bin_{bins-1}],
            "n_wrong": int,
            "n_high_conf_wrong_0p9": int,   # confidence >= 0.9 among wrong
            "n_high_conf_wrong_0p95": int,
            "mean_wrong_confidence": float | None,
        }
    """
    probabilities = probabilities.detach().cpu().float()
    predictions = _ensure_tensor_long(predictions)
    labels = _ensure_tensor_long(labels)
    confidences = probabilities.max(dim=1).values
    wrong_mask = predictions != labels
    n_wrong = int(wrong_mask.sum().item())
    if n_wrong == 0:
        return {
            "bin_edges": [round(i / bins, 6) for i in range(bins + 1)],
            "counts": [0] * bins,
            "n_wrong": 0,
            "n_high_conf_wrong_0p9": 0,
            "n_high_conf_wrong_0p95": 0,
            "mean_wrong_confidence": None,
        }
    wrong_conf = confidences[wrong_mask]
    hist = torch.histc(wrong_conf, bins=bins, min=0.0, max=1.0)
    bin_edges = [round(i / bins, 6) for i in range(bins + 1)]
    return {
        "bin_edges": bin_edges,
        "counts": [int(c) for c in hist.tolist()],
        "n_wrong": n_wrong,
        "n_high_conf_wrong_0p9": int((wrong_conf >= 0.9).sum().item()),
        "n_high_conf_wrong_0p95": int((wrong_conf >= 0.95).sum().item()),
        "mean_wrong_confidence": float(wrong_conf.mean().item()),
    }


def per_fold_accuracy(fold_summaries: list[dict]) -> list[dict]:
    """Convenience: from aggregate_summary['folds'] extract per-fold accuracies."""
    rows = []
    for entry in fold_summaries:
        rows.append({
            "seed": entry.get("seed"),
            "fold_index": entry.get("fold_index"),
            "best_val_accuracy": entry.get("best_val_accuracy"),
            "best_epoch": entry.get("best_epoch"),
        })
    return rows


def class_1_bias_summary(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
) -> dict:
    """Compact bias scorecard around class 1 — the key TestA failure mode.

    Always uses class 1 as target (the project's identified bias). Returns the
    metrics that the user explicitly asked us to report:
        - class_1_overprediction_ratio (predicted_count[1] / true_count[1])
        - class_8_accuracy (since class 8 is the most-suppressed in baseline)
        - total_x_to_1_errors
        - pair_errors for SYSTEMATIC_TARGETS_TO_ONE
    """
    predictions = _ensure_tensor_long(predictions)
    labels = _ensure_tensor_long(labels)
    return {
        "class_1_overprediction_ratio": class_overprediction_ratio(predictions, labels, target_class=1),
        "class_1_accuracy": class_accuracy(predictions, labels, target_class=1),
        "class_8_accuracy": class_accuracy(predictions, labels, target_class=8),
        **x_to_target_errors(predictions, labels, target_class=1, num_classes=num_classes),
    }


def save_extended_diagnostics(
    output_dir: Path,
    sample_ids: list[int | str],
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    fold_summaries: list[dict] | None = None,
    baseline_predictions: torch.Tensor | None = None,
    baseline_sample_ids: list[int | str] | None = None,
    baseline_labels: torch.Tensor | None = None,
    baseline_experiment_id: str | None = None,
    num_classes: int = 10,
) -> dict:
    """Compute the FULL extended diagnostics payload and write it as JSON.

    Writes:
        <output_dir>/extended_diagnostics.json

    The JSON contains everything the user asked for in section 七 of the plan:
        - overall accuracy
        - per-class accuracy
        - confusion matrix counts
        - true_count_by_class / predicted_count_by_class / ratio_by_class
        - class_1_overprediction_ratio
        - class_8_accuracy
        - total_x_to_1_errors + per-pair X→1 breakdown
        - top confused pairs
        - best_epoch_by_fold + fold-level accuracy (if fold_summaries supplied)
        - confidence_distribution_wrong (including high-conf wrong tail)
        - error_overlap_with_baseline (if baseline supplied)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _ensure_tensor_long(labels)
    probabilities = probabilities.detach().cpu().float()
    predictions = probabilities.argmax(dim=1)

    overall_accuracy = float((predictions == labels).float().mean().item()) if labels.numel() else 0.0
    per_class = per_class_accuracy(labels, predictions, num_classes=num_classes)
    confused = top_confused_pairs(labels, predictions, num_classes=num_classes, top_k=15)

    true_counts = [int((labels == c).sum().item()) for c in range(num_classes)]
    pred_counts = [int((predictions == c).sum().item()) for c in range(num_classes)]
    ratios = [(pred_counts[c] / true_counts[c]) if true_counts[c] else None for c in range(num_classes)]

    bias = class_1_bias_summary(predictions, labels, num_classes=num_classes)
    conf_dist = confidence_distribution_wrong(probabilities, predictions, labels)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(labels.numpy(), predictions.numpy()):
        cm[int(true_label), int(predicted_label)] += 1

    high_conf_wrong = high_confidence_wrong_samples(sample_ids, probabilities, predictions, labels, top_k=50)

    payload: dict = {
        "overall_accuracy": overall_accuracy,
        "num_samples": int(labels.numel()),
        "per_class_accuracy": per_class,
        "true_count_by_class": true_counts,
        "predicted_count_by_class": pred_counts,
        "predicted_true_ratio_by_class": ratios,
        "class_1_overprediction_ratio": bias["class_1_overprediction_ratio"],
        "class_1_accuracy": bias["class_1_accuracy"],
        "class_8_accuracy": bias["class_8_accuracy"],
        "total_x_to_1_errors": bias["total_x_to_target"],
        "x_to_1_pair_errors": bias["pair_errors"],
        "systematic_x_to_1_pair_errors": bias["systematic_pair_errors"],
        "top_confused_pairs": confused,
        "confusion_matrix": cm.tolist(),
        "confidence_distribution_wrong": conf_dist,
        "high_confidence_wrong_samples": high_conf_wrong,
    }

    if fold_summaries is not None:
        payload["fold_accuracy"] = per_fold_accuracy(fold_summaries)
        best_epochs = [row["best_epoch"] for row in fold_summaries if row.get("best_epoch") is not None]
        if best_epochs:
            payload["best_epoch_mean"] = float(sum(best_epochs) / len(best_epochs))
            payload["best_epoch_max"] = int(max(best_epochs))
            payload["best_epoch_min"] = int(min(best_epochs))

    if baseline_predictions is not None and baseline_sample_ids is not None and baseline_labels is not None:
        overlap = error_overlap(
            sample_ids,
            predictions,
            labels,
            baseline_sample_ids,
            baseline_predictions,
            baseline_labels,
            target_class=1,
        )
        overlap["baseline_experiment_id"] = baseline_experiment_id
        payload["error_overlap_with_baseline"] = overlap

    (output_dir / "extended_diagnostics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload
