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
