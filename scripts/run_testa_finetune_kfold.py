from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_config import (
    assert_safe_output_dir,
    config_list,
    config_section,
    load_experiment_config,
    resolve_experiment_output_dir,
    resolve_path,
    write_config_snapshot,
)
from src.testa_robust_train import TestARobustConfig, train as train_fold


DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "testa_finetune_from_generalist.yaml"


def parse_args(default_config: Path = DEFAULT_CONFIG):
    parser = argparse.ArgumentParser(description="Run strict TestA-only specialist K-Fold fine-tuning.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=None, help="Exclusive end fold.")
    parser.add_argument("--allow-outputs-submission", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Optional quick smoke-test override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional quick smoke-test override.")
    return parser.parse_args()


def build_fold_config(raw_config: dict, project_root: Path, output_base: Path, seed: int, fold_index: int, n_splits: int, args) -> TestARobustConfig:
    model = config_section(raw_config, "model")
    training = config_section(raw_config, "training")
    augmentation = config_section(raw_config, "augmentation")
    mode = str(training.get("mode", "testa_finetune"))
    init_checkpoint = None if mode == "testa_scratch" else resolve_path(model.get("init_checkpoint"), project_root)
    fold_dir = output_base / f"seed_{seed}" / f"fold_{fold_index}"
    return TestARobustConfig(
        project_root=project_root,
        output_dir=fold_dir,
        base_checkpoint=init_checkpoint,
        model_name=str(model.get("model_name", "medium_cnn")),
        dropout=float(model.get("dropout", 0.21672530847241062)),
        seed=int(seed),
        epochs=int(args.epochs if args.epochs is not None else training.get("epochs", 20)),
        batch_size=int(args.batch_size if args.batch_size is not None else training.get("batch_size", 256)),
        learning_rate=float(training.get("learning_rate", 3e-5)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        label_smoothing=float(training.get("label_smoothing", 0.02)),
        patience=int(training.get("patience", 5)),
        use_amp=bool(training.get("use_amp", True)),
        allow_tf32=bool(training.get("allow_tf32", True)),
        num_workers=int(training.get("num_workers", 0)),
        use_kfold=True,
        kfold_n_splits=int(n_splits),
        kfold_index=int(fold_index),
        mixup_alpha=float(augmentation.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(augmentation.get("cutmix_alpha", 0.0)),
        mix_prob=float(augmentation.get("mix_prob", 0.0)),
        random_erasing_p=float(augmentation.get("random_erasing_p", 0.0)),
        checkpoint_name=str(training.get("checkpoint_name", "testa_specialist_best.pt")),
        training_mode=mode,
        experiment_id=str(raw_config["experiment_id"]),
        preprocessing_mode=str(augmentation.get("preprocessing_mode", "raw")),
        evaluate_preprocess=bool(augmentation.get("evaluate_preprocess", False)),
        freeze_backbone_epochs=int(training.get("freeze_backbone_epochs", 0)),
        affine_degrees=float(augmentation.get("affine_degrees", 5.0)),
        translate_ratio=float(augmentation.get("translate_ratio", 0.04)),
        scale_min=float(augmentation.get("scale_min", 0.96)),
        scale_max=float(augmentation.get("scale_max", 1.04)),
        shear_degrees=float(augmentation.get("shear_degrees", 2.0)),
        use_testa_like_augment=bool(augmentation.get("use_testa_like_augment", False)),
        preprocess_probability=float(augmentation.get("preprocess_probability", 0.0)),
        save_validation_artifacts=True,
    )


def run_from_config(default_config: Path = DEFAULT_CONFIG, forced_mode: str | None = None):
    args = parse_args(default_config)
    project_root = args.project_root.resolve()
    raw_config = load_experiment_config(args.config)
    if forced_mode is not None:
        raw_config.setdefault("training", {})["mode"] = forced_mode
        if forced_mode == "testa_scratch":
            raw_config.setdefault("model", {})["init_checkpoint"] = None
    output_base = resolve_experiment_output_dir(raw_config, project_root)
    assert_safe_output_dir(output_base, project_root, allow_outputs_submission=args.allow_outputs_submission)
    output_base.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(raw_config, output_base)

    seeds = [int(seed) for seed in config_list(raw_config, "seeds", [raw_config.get("seed", 42)])]
    n_splits = int(raw_config.get("folds", config_section(raw_config, "data").get("num_folds", 5)))
    end_fold = args.end_fold if args.end_fold is not None else n_splits
    fold_summaries = []
    started = time.perf_counter()

    for seed in seeds:
        for fold_index in range(args.start_fold, end_fold):
            fold_started = time.perf_counter()
            print(f"\n{'=' * 72}\n[TestA specialist] seed={seed} fold={fold_index}/{n_splits - 1}\n{'=' * 72}", flush=True)
            config = build_fold_config(raw_config, project_root, output_base, seed, fold_index, n_splits, args)
            manifest = train_fold(config)
            fold_summaries.append({
                "seed": seed,
                "fold_index": fold_index,
                "checkpoint": manifest["checkpoint"],
                "best_val_accuracy": manifest["best_val_accuracy"],
                "best_epoch": manifest["best_epoch"],
                "output_dir": manifest["output_dir"],
                "elapsed_sec": round(time.perf_counter() - fold_started, 2),
            })

    summary = {
        "experiment_id": raw_config["experiment_id"],
        "config": str(args.config),
        "output_dir": str(output_base),
        "seeds": seeds,
        "n_splits": n_splits,
        "folds": fold_summaries,
        "mean_best_val_accuracy": (sum(row["best_val_accuracy"] for row in fold_summaries) / len(fold_summaries)) if fold_summaries else None,
        "total_elapsed_sec": round(time.perf_counter() - started, 2),
    }
    summary_path = output_base / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[TestA specialist] summary: {summary_path}")
    return summary


if __name__ == "__main__":
    run_from_config()
