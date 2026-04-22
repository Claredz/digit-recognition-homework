import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.config import ExperimentConfig, ensure_project_paths
from src.data import create_dataloaders
from src.engine import fit
from src.model import SmallCNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="训练手写数字 CNN 基线模型")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-name", choices=["mnist", "folder"], default="mnist")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        project_root=args.project_root.resolve(),
        dataset_name=args.dataset_name,
        data_dir=args.data_dir.resolve() if args.data_dir is not None else None,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )

    set_seed(config.seed)
    paths = ensure_project_paths(config)
    train_loader, val_loader = create_dataloaders(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    history = fit(model, train_loader, val_loader, config=config, paths=paths, device=device)

    run_manifest = {
        "dataset_name": config.dataset_name,
        "data_dir": str(config.resolved_data_dir()),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "best_val_accuracy": history["best_val_accuracy"],
        "checkpoint": str(paths.checkpoints_dir / "best_model.pt"),
    }
    (paths.logs_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

    print(f"训练完成。验证集最佳准确率：{history['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
