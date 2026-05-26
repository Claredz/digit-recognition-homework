from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.error_analysis import save_error_analysis_bundle, save_extended_diagnostics
from src.evaluate import save_evaluation_bundle
from src.experiment_config import (
    assert_safe_output_dir,
    config_list,
    load_experiment_config,
    resolve_experiment_output_dir,
    validate_experiment_config,
)
from src.testa_robust_train import IdxTestADataset


DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "testa_specialist_5fold.yaml"
DEFAULT_BASELINE_EXPERIMENT_ID = "testa_partial_init_lr1e4_mixup01_erasing005_e40"


def parse_args():
    parser = argparse.ArgumentParser(description="Build TestA specialist out-of-fold evaluation artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--allow-partial", action="store_true", help="Allow missing OOF samples for smoke runs.")
    parser.add_argument("--allow-outputs-submission", action="store_true")
    parser.add_argument(
        "--baseline-experiment-id",
        type=str,
        default=DEFAULT_BASELINE_EXPERIMENT_ID,
        help=(
            "baseline experiment_id to compute error_overlap against; default = "
            f"{DEFAULT_BASELINE_EXPERIMENT_ID}. Set to 'none' or '' to skip overlap."
        ),
    )
    return parser.parse_args()


def load_fold_probabilities(fold_dir: Path):
    path = fold_dir / "predictions" / "validation_probabilities.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    return payload["sample_ids"], payload["labels"].cpu().long(), payload["probabilities"].cpu().float(), str(path)


def _maybe_load_baseline(project_root: Path, experiment_id: str | None):
    """Try to load an existing baseline OOF predictions for overlap comparison.

    Returns (sample_ids, labels_tensor, predictions_tensor) or (None, None, None)
    when the baseline is absent or explicitly disabled.
    """
    if not experiment_id or experiment_id.lower() == "none":
        return None, None, None
    baseline_oof = project_root / "outputs_runs" / experiment_id / "oof" / "oof_probabilities.pt"
    if not baseline_oof.exists():
        return None, None, None
    payload = torch.load(baseline_oof, map_location="cpu", weights_only=False)
    sample_ids = [int(sid) for sid in payload["sample_ids"]]
    labels = payload["labels"].cpu().long()
    probabilities = payload["probabilities"].cpu().float()
    predictions = probabilities.argmax(dim=1)
    return sample_ids, labels, predictions


def _maybe_load_fold_summaries(output_base: Path) -> list[dict]:
    """Read aggregate_summary.json so extended diagnostics can include best-epoch info."""
    aggregate_path = output_base / "aggregate_summary.json"
    if not aggregate_path.exists():
        return []
    try:
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
        folds = payload.get("folds", [])
        if isinstance(folds, list):
            return folds
    except Exception:
        return []
    return []


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_config = load_experiment_config(args.config)
    validate_experiment_config(raw_config, strict=False)
    output_base = resolve_experiment_output_dir(raw_config, project_root)
    assert_safe_output_dir(output_base, project_root, allow_outputs_submission=args.allow_outputs_submission)
    seeds = [int(seed) for seed in config_list(raw_config, "seeds", [raw_config.get("seed", 42)])]
    n_splits = int(raw_config.get("folds", 5))

    sample_probabilities: dict[int, list[torch.Tensor]] = {}
    sample_labels: dict[int, int] = {}
    sources = []
    for seed in seeds:
        for fold_index in range(n_splits):
            fold_dir = output_base / f"seed_{seed}" / f"fold_{fold_index}"
            sample_ids, labels, probabilities, source_path = load_fold_probabilities(fold_dir)
            for local_index, sample_id in enumerate(sample_ids):
                numeric_id = int(sample_id)
                label = int(labels[local_index].item())
                if numeric_id in sample_labels and sample_labels[numeric_id] != label:
                    raise RuntimeError(f"sample_id={numeric_id} label mismatch across folds/seeds")
                sample_labels[numeric_id] = label
                sample_probabilities.setdefault(numeric_id, []).append(probabilities[local_index])
            sources.append({"seed": seed, "fold_index": fold_index, "path": source_path, "n": len(sample_ids)})

    ordered_ids = sorted(sample_probabilities)
    image_path = project_root / "data" / "test_A_images.idx3-ubyte(1)" / "test_A_images.idx3-ubyte"
    label_path = project_root / "data" / "test_A_labels.idx1-ubyte(1)" / "test_A_labels.idx1-ubyte"
    full_dataset = IdxTestADataset(image_path, label_path, preprocess=False)
    if not args.allow_partial and len(ordered_ids) != len(full_dataset):
        raise RuntimeError(f"OOF 覆盖不完整: {len(ordered_ids)} / {len(full_dataset)}")

    probabilities = torch.stack([torch.stack(sample_probabilities[sample_id], dim=0).mean(dim=0) for sample_id in ordered_ids])
    labels = torch.tensor([sample_labels[sample_id] for sample_id in ordered_ids], dtype=torch.long)
    predictions = probabilities.argmax(dim=1)
    images = torch.stack([full_dataset[int(sample_id)][0] for sample_id in ordered_ids])
    accuracy = float((predictions == labels).float().mean().item()) if labels.numel() else 0.0

    oof_dir = output_base / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    probability_path = oof_dir / "oof_probabilities.pt"
    torch.save({"sample_ids": ordered_ids, "labels": labels, "probabilities": probabilities}, probability_path)

    csv_path = oof_dir / "oof_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "label", "prediction", "confidence", "correct"] + [f"prob_{index}" for index in range(10)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        confidences = probabilities.max(dim=1).values
        for row_index, sample_id in enumerate(ordered_ids):
            writer.writerow({
                "sample_id": sample_id,
                "label": int(labels[row_index].item()),
                "prediction": int(predictions[row_index].item()),
                "confidence": float(confidences[row_index].item()),
                "correct": bool(predictions[row_index].item() == labels[row_index].item()),
                **{f"prob_{class_index}": float(probabilities[row_index, class_index].item()) for class_index in range(10)},
            })

    summary = save_evaluation_bundle(images, labels, predictions, oof_dir)
    error_analysis = save_error_analysis_bundle(oof_dir, ordered_ids, images, labels, probabilities)

    # Phase A.2b: extended diagnostics (class-1 bias, X→1 errors, baseline overlap, confidence dist)
    baseline_id = args.baseline_experiment_id
    if baseline_id == raw_config["experiment_id"]:
        # avoid trivial self-overlap (would report 100%)
        baseline_id = None
    baseline_ids, baseline_labels, baseline_predictions = _maybe_load_baseline(project_root, baseline_id)
    fold_summaries = _maybe_load_fold_summaries(output_base)

    extended = save_extended_diagnostics(
        oof_dir,
        ordered_ids,
        labels,
        probabilities,
        fold_summaries=fold_summaries if fold_summaries else None,
        baseline_predictions=baseline_predictions,
        baseline_sample_ids=baseline_ids,
        baseline_labels=baseline_labels,
        baseline_experiment_id=baseline_id if baseline_predictions is not None else None,
    )

    payload = {
        "experiment_id": raw_config["experiment_id"],
        "accuracy": accuracy,
        "num_samples": int(labels.numel()),
        "sources": sources,
        "oof_probabilities": str(probability_path),
        "oof_predictions": str(csv_path),
        "summary": summary,
        "error_analysis": error_analysis,
        "extended_diagnostics_path": str(oof_dir / "extended_diagnostics.json"),
        "class_1_overprediction_ratio": extended.get("class_1_overprediction_ratio"),
        "class_8_accuracy": extended.get("class_8_accuracy"),
        "total_x_to_1_errors": extended.get("total_x_to_1_errors"),
        "baseline_experiment_id": baseline_id if baseline_predictions is not None else None,
    }
    (oof_dir / "oof_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "accuracy": accuracy,
        "num_samples": int(labels.numel()),
        "class_1_overprediction_ratio": extended.get("class_1_overprediction_ratio"),
        "class_8_accuracy": extended.get("class_8_accuracy"),
        "total_x_to_1_errors": extended.get("total_x_to_1_errors"),
        "output_dir": str(oof_dir),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
