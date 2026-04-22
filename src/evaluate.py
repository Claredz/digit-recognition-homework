import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import ExperimentConfig, ensure_project_paths
from src.data import create_dataloaders
from src.model import SmallCNN


def save_evaluation_bundle(
    images: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    output_dir: Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = float(accuracy_score(y_true.numpy(), y_pred.numpy()))
    matrix = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=list(range(10)))

    (output_dir / "summary.json").write_text(
        json.dumps({"accuracy": accuracy}, indent=2),
        encoding="utf-8",
    )
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title("混淆矩阵")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    wrong_indices = (y_true != y_pred).nonzero(as_tuple=False).flatten().tolist()
    if not wrong_indices:
        wrong_indices = [0]

    fig, axes = plt.subplots(1, min(4, len(wrong_indices)), figsize=(10, 3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for axis, index in zip(axes, wrong_indices[:4]):
        axis.imshow(images[index].squeeze(0).numpy(), cmap="gray")
        axis.set_title(f"T:{int(y_true[index])} P:{int(y_pred[index])}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "misclassified_grid.png")
    plt.close(fig)


def collect_predictions(model, loader, device: str):
    model.eval()
    image_batches = []
    true_batches = []
    pred_batches = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            image_batches.append(images.cpu())
            true_batches.append(labels.cpu())
            pred_batches.append(predictions)

    return torch.cat(image_batches), torch.cat(true_batches), torch.cat(pred_batches)


def parse_args():
    parser = argparse.ArgumentParser(description="评估已训练的手写数字模型")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-name", choices=["mnist", "folder"], default="mnist")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=28)
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
    )
    paths = ensure_project_paths(config)
    _, val_loader = create_dataloaders(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SmallCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    images, y_true, y_pred = collect_predictions(model, val_loader, device=device)
    save_evaluation_bundle(images, y_true, y_pred, paths.figures_dir)
    print(f"评估完成。准确率已保存到 {paths.figures_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
