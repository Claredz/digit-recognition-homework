"""Drive 5-fold TestA fine-tuning runs sequentially.

Each fold trains an independent checkpoint with:
- 80% TestA as training-pool ingredient (via kfold_indices)
- 20% TestA as heldout validation (the fold)
- Same base checkpoint, same hyperparameters
- MixUp + CutMix + RandomErasing on the TestA training samples
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.testa_robust_train import TestARobustConfig, train as train_fold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "outputs_submission" / "checkpoints" / "robust_expert_v2_testa_partial_best_epoch12_score07098.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs_submission")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--epoch-size", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--clean-weight", type=float, default=0.25)
    parser.add_argument("--mnist-c-weight", type=float, default=0.25)
    parser.add_argument("--synthetic-weight", type=float, default=0.20)
    parser.add_argument("--testa-weight", type=float, default=0.30)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=0.5)
    parser.add_argument("--random-erasing-p", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=None, help="Exclusive end. If None, runs to n-splits.")
    parser.add_argument("--run-name", default="testa_kfold")
    return parser.parse_args()


def archive_log(logs_dir: Path, run_name: str, fold_index: int):
    """Copy the per-fold history/manifest with a fold suffix."""
    moved = {}
    for stem in ("testa_robust_v2_history.csv", "testa_robust_v2_history.json", "testa_robust_v2_manifest.json", "testa_robust_v2_data_manifest.json"):
        src = logs_dir / stem
        if not src.exists():
            continue
        dst = logs_dir / f"{run_name}_fold{fold_index}_{stem}"
        shutil.copyfile(src, dst)
        moved[stem] = str(dst)
    return moved


def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    end_fold = args.end_fold if args.end_fold is not None else args.n_splits
    fold_summaries = []
    overall_start = time.perf_counter()

    for fold_index in range(args.start_fold, end_fold):
        fold_start = time.perf_counter()
        checkpoint_name = f"robust_expert_v2_kfold_f{fold_index}_best.pt"
        print(f"\n{'='*60}\n[kfold] fold {fold_index}/{args.n_splits-1} -> {checkpoint_name}\n{'='*60}", flush=True)

        config = TestARobustConfig(
            project_root=PROJECT_ROOT.resolve(),
            output_dir=output_dir,
            base_checkpoint=args.base_checkpoint.resolve(),
            epochs=args.epochs,
            epoch_size=args.epoch_size,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            label_smoothing=args.label_smoothing,
            clean_weight=args.clean_weight,
            mnist_c_weight=args.mnist_c_weight,
            synthetic_weight=args.synthetic_weight,
            use_testa_partial=True,
            testa_weight=args.testa_weight,
            use_kfold=True,
            kfold_n_splits=args.n_splits,
            kfold_index=fold_index,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
            random_erasing_p=args.random_erasing_p,
            num_workers=args.num_workers,
            patience=args.patience,
            checkpoint_name=checkpoint_name,
        )

        manifest = train_fold(config)
        elapsed = round(time.perf_counter() - fold_start, 1)
        archived = archive_log(logs_dir, args.run_name, fold_index)
        fold_summaries.append({
            "fold_index": fold_index,
            "checkpoint": manifest["checkpoint"],
            "best_score": manifest["best_score"],
            "best_epoch": manifest["best_epoch"],
            "elapsed_sec": elapsed,
            "archived_logs": archived,
        })
        print(f"[kfold] fold {fold_index} done score={manifest['best_score']:.4f} epoch={manifest['best_epoch']} elapsed={elapsed}s", flush=True)

    total_elapsed = round(time.perf_counter() - overall_start, 1)
    summary_path = logs_dir / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps({
        "run_name": args.run_name,
        "base_checkpoint": str(args.base_checkpoint),
        "n_splits": args.n_splits,
        "start_fold": args.start_fold,
        "end_fold": end_fold,
        "epochs_per_fold": args.epochs,
        "epoch_size": args.epoch_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "testa_weight": args.testa_weight,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "mix_prob": args.mix_prob,
        "random_erasing_p": args.random_erasing_p,
        "folds": fold_summaries,
        "total_elapsed_sec": total_elapsed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[kfold] all folds done. summary saved to {summary_path}. total {total_elapsed}s", flush=True)
    print("=== per-fold best score ===")
    for fold in fold_summaries:
        print(f"  fold {fold['fold_index']}: score={fold['best_score']:.4f} epoch={fold['best_epoch']} t={fold['elapsed_sec']}s")
    if fold_summaries:
        mean_score = sum(f["best_score"] for f in fold_summaries) / len(fold_summaries)
        print(f"  mean heldout score: {mean_score:.4f}")


if __name__ == "__main__":
    main()
