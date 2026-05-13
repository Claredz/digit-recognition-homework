import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import ExperimentConfig, ensure_project_paths
from src.data import create_dataloaders
from src.model import build_model


def _configure_matplotlib_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def checkpoint_state_dict(checkpoint_payload):
    if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload:
        return checkpoint_payload["model_state_dict"]
    return checkpoint_payload


def load_model_from_checkpoint(checkpoint_path: Path, config: ExperimentConfig, device: str):
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint_payload.get("config", {}) if isinstance(checkpoint_payload, dict) else {}
    model_name = checkpoint_payload.get("model_name") if isinstance(checkpoint_payload, dict) else None
    model_name = model_name or checkpoint_config.get("model_name", config.model_name)
    dropout = checkpoint_config.get("dropout", config.dropout) if isinstance(checkpoint_config, dict) else config.dropout
    model = build_model(model_name, num_classes=config.num_classes, in_channels=config.in_channels, dropout=dropout)
    model.load_state_dict(checkpoint_state_dict(checkpoint_payload))
    model.to(device)
    model.eval()
    return model, checkpoint_payload


def save_evaluation_bundle(
    images: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    output_dir: Path,
    num_classes: int = 10,
):
    _configure_matplotlib_chinese_font()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    accuracy = float(accuracy_score(y_true_np, y_pred_np))
    labels = list(range(num_classes))
    matrix = confusion_matrix(y_true_np, y_pred_np, labels=labels)
    report = classification_report(y_true_np, y_pred_np, labels=labels, output_dict=True, zero_division=0)
    summary = {"accuracy": accuracy, "num_samples": int(len(y_true_np))}

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("混淆矩阵（计数）")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            value = int(matrix[i, j])
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    wrong_indices = (y_true != y_pred).nonzero(as_tuple=False).flatten().tolist()
    if not wrong_indices:
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.text(0.5, 0.5, "无误分类样本", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "misclassified_grid.png")
        plt.close(fig)
        return summary

    fig, axes = plt.subplots(1, min(4, len(wrong_indices)), figsize=(10, 3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for axis, index in zip(axes, wrong_indices[:4]):
        image = images[index]
        if torch.is_tensor(image):
            image = (image * 0.5 + 0.5).clamp(0, 1)
            image = image.squeeze(0).numpy()
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"真:{int(y_true[index])} 预测:{int(y_pred[index])}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "misclassified_grid.png")
    plt.close(fig)
    return summary


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
    parser.add_argument("--dataset-name", choices=["mnist", "folder", "multisource", "submission"], default="mnist")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="small_cnn")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        project_root=args.project_root.resolve(),
        dataset_name=args.dataset_name,
        data_dir=args.data_dir.resolve() if args.data_dir is not None else None,
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        model_name=args.model_name,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        image_size=args.image_size,
    )
    paths = ensure_project_paths(config)
    _, val_loader = create_dataloaders(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model_from_checkpoint(args.checkpoint, config, device)

    images, y_true, y_pred = collect_predictions(model, val_loader, device=device)
    summary = save_evaluation_bundle(images, y_true, y_pred, paths.evaluation_dir, num_classes=config.num_classes)
    print(f"评估完成。accuracy={summary['accuracy']:.4f}，结果已保存到 {paths.evaluation_dir}")


if __name__ == "__main__":
    main()
