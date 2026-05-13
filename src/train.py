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
    )

    set_seed(config.seed)
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
        "checkpoint": str(paths.checkpoints_dir / "best_model.pt"),
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
