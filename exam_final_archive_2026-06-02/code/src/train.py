import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.config import ExperimentConfig, ensure_project_paths
from src.data import create_dataloaders
from src.engine import fit
from src.model import build_model, count_model_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="训练手写数字 CNN 模型")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-name", choices=["mnist", "folder", "multisource", "submission"], default="mnist")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-name", default="small_cnn")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--optimizer-type", choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument("--scheduler-type", default="none")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-emnist-digits", action="store_true")
    parser.add_argument("--use-usps", action="store_true")
    parser.add_argument("--use-qmnist", action="store_true")
    parser.add_argument("--use-early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--training-mode", choices=["clean", "robust_finetune"], default="clean")
    parser.add_argument("--clean-checkpoint-path", type=Path, default=None)
    parser.add_argument("--robust-checkpoint-path", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-4)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--freeze-backbone-first", action="store_true")
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--robust-aug-strength", choices=["light", "medium", "strong"], default="medium")
    parser.add_argument("--no-robust-sampler", action="store_true")
    parser.add_argument("--use-local-digits", action="store_true")
    parser.add_argument("--local-digits-dir", type=Path, default=None)
    parser.add_argument("--local-digits-holdout-dir", type=Path, default=None)
    parser.add_argument("--use-hasyv2", action="store_true")
    parser.add_argument("--hasyv2-dir", type=Path, default=None)
    parser.add_argument("--use-chars74k", action="store_true")
    parser.add_argument("--chars74k-dir", type=Path, default=None)
    parser.add_argument("--use-penbased-rendered", action="store_true")
    parser.add_argument("--penbased-dir", type=Path, default=None)
    parser.add_argument("--use-optical-digits", action="store_true")
    parser.add_argument("--optical-dir", type=Path, default=None)
    parser.add_argument("--mnist-family-weight", type=float, default=0.60)
    parser.add_argument("--local-digits-weight", type=float, default=0.20)
    parser.add_argument("--hasyv2-weight", type=float, default=0.10)
    parser.add_argument("--chars74k-weight", type=float, default=0.05)
    parser.add_argument("--penbased-weight", type=float, default=0.08)
    parser.add_argument("--optical-weight", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        project_root=args.project_root.resolve(),
        dataset_name=args.dataset_name,
        data_dir=args.data_dir.resolve() if args.data_dir is not None else None,
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        run_name=args.run_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        dropout=args.dropout,
        optimizer_type=args.optimizer_type,
        scheduler_type=args.scheduler_type,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        max_samples=args.max_samples,
        use_emnist_digits=args.use_emnist_digits,
        use_usps=args.use_usps,
        use_qmnist=args.use_qmnist,
        use_early_stopping=args.use_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        training_mode=args.training_mode,
        clean_checkpoint_path=args.clean_checkpoint_path.resolve() if args.clean_checkpoint_path is not None else Path("outputs_submission/checkpoints/checkpoint_clean_best.pth"),
        robust_checkpoint_path=args.robust_checkpoint_path.resolve() if args.robust_checkpoint_path is not None else None,
        checkpoint_name=args.checkpoint_name,
        fine_tune_lr=args.finetune_learning_rate,
        fine_tune_epochs=args.finetune_epochs,
        freeze_backbone_first=args.freeze_backbone_first,
        freeze_epochs=args.freeze_epochs,
        robust_aug_strength=args.robust_aug_strength,
        use_robust_sampler=not args.no_robust_sampler,
        use_local_digits=args.use_local_digits,
        local_digits_dir=args.local_digits_dir.resolve() if args.local_digits_dir is not None else None,
        local_digits_holdout_dir=args.local_digits_holdout_dir.resolve() if args.local_digits_holdout_dir is not None else None,
        use_hasyv2=args.use_hasyv2,
        hasyv2_dir=args.hasyv2_dir.resolve() if args.hasyv2_dir is not None else None,
        use_chars74k=args.use_chars74k,
        chars74k_dir=args.chars74k_dir.resolve() if args.chars74k_dir is not None else None,
        use_penbased_rendered=args.use_penbased_rendered,
        penbased_dir=args.penbased_dir.resolve() if args.penbased_dir is not None else None,
        use_optical_digits=args.use_optical_digits,
        optical_dir=args.optical_dir.resolve() if args.optical_dir is not None else None,
        mnist_family_weight=args.mnist_family_weight,
        local_digits_weight=args.local_digits_weight,
        hasyv2_weight=args.hasyv2_weight,
        chars74k_weight=args.chars74k_weight,
        penbased_weight=args.penbased_weight,
        optical_weight=args.optical_weight,
    )

    set_seed(config.seed)
    if config.training_mode == "robust_finetune":
        from src.robust_train import run_robust_finetune

        run_robust_finetune(config)
        return

    paths = ensure_project_paths(config)
    train_loader, val_loader = create_dataloaders(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config)
    total_params, trainable_params = count_model_parameters(model)
    history = fit(model, train_loader, val_loader, config=config, paths=paths, device=device)

    run_manifest = {
        "dataset_name": config.dataset_name,
        "data_dir": str(config.resolved_data_dir()),
        "output_dir": str(paths.outputs_dir),
        "model_name": config.model_name,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "best_val_accuracy": history["best_val_accuracy"],
        "best_epoch": history["best_epoch"],
        "checkpoint": str(paths.checkpoints_dir / config.checkpoint_name),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    (paths.logs_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

    print(f"训练完成。验证集最佳准确率：{history['best_val_accuracy']:.4f}")
    print(f"输出目录：{paths.outputs_dir}")


if __name__ == "__main__":
    main()
