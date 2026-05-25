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

from src.error_analysis import save_error_analysis_bundle
from src.evaluate import save_evaluation_bundle
from src.experiment_config import assert_safe_output_dir, config_list, load_experiment_config, resolve_experiment_output_dir
from src.testa_robust_train import IdxTestADataset


DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "testa_specialist_5fold.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Build TestA specialist out-of-fold evaluation artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--allow-partial", action="store_true", help="Allow missing OOF samples for smoke runs.")
    parser.add_argument("--allow-outputs-submission", action="store_true")
    return parser.parse_args()


def load_fold_probabilities(fold_dir: Path):
    path = fold_dir / "predictions" / "validation_probabilities.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    return payload["sample_ids"], payload["labels"].cpu().long(), payload["probabilities"].cpu().float(), str(path)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_config = load_experiment_config(args.config)
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
    payload = {
        "experiment_id": raw_config["experiment_id"],
        "accuracy": accuracy,
        "num_samples": int(labels.numel()),
        "sources": sources,
        "oof_probabilities": str(probability_path),
        "oof_predictions": str(csv_path),
        "summary": summary,
        "error_analysis": error_analysis,
    }
    (oof_dir / "oof_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"accuracy": accuracy, "num_samples": int(labels.numel()), "output_dir": str(oof_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
