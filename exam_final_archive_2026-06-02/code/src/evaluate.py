import argparse
import csv
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config import ExperimentConfig, ensure_project_paths
from src.data import build_eval_transform, create_dataloaders
from src.model import build_model


def _configure_matplotlib_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def checkpoint_state_dict(checkpoint_payload):
    if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload:
        return checkpoint_payload["model_state_dict"]
    return checkpoint_payload


def load_model_from_checkpoint(checkpoint_path: Path, config: ExperimentConfig, device: str):
    if config.verbose:
        print(f"[load] checkpoint={checkpoint_path} device={device}", flush=True)
    checkpoint_payload = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint_payload.get("config", {}) if isinstance(checkpoint_payload, dict) else {}
    checkpoint_model_name = checkpoint_payload.get("model_name") if isinstance(checkpoint_payload, dict) else None
    checkpoint_model_name = checkpoint_model_name or checkpoint_config.get("model_name", config.model_name)
    dropout = checkpoint_config.get("dropout", config.dropout) if isinstance(checkpoint_config, dict) else config.dropout

    # Try the config's model_name first (supports heterogeneous fine-tuning),
    # fall back to the checkpoint's model_name for backward compatibility.
    for model_name in (config.model_name, checkpoint_model_name):
        try:
            model = build_model(model_name, num_classes=config.num_classes, in_channels=config.in_channels, dropout=dropout)
            source_state = checkpoint_state_dict(checkpoint_payload)
            current_state = model.state_dict()
            compatible = {k: v for k, v in source_state.items() if k in current_state and current_state[k].shape == v.shape}
            if compatible:
                current_state.update(compatible)
                model.load_state_dict(current_state)
                model.to(device)
                model.eval()
                if config.verbose:
                    print(f"[load] model={model_name} matched={len(compatible)}/{len(current_state)}", flush=True)
                return model, checkpoint_payload
        except Exception:
            continue

    raise RuntimeError(f"Failed to load checkpoint {checkpoint_path} with any model")


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


def collect_predictions(model, loader, device: str, config: ExperimentConfig | None = None, stage_name: str = "evaluate"):
    model.eval()
    image_batches = []
    true_batches = []
    pred_batches = []
    verbose = True if config is None else config.verbose
    log_interval = max(1, 50 if config is None else config.log_interval)
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            image_batches.append(images.cpu())
            true_batches.append(labels.cpu())
            pred_batches.append(predictions)
            if verbose and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches):
                total_text = str(total_batches) if total_batches is not None else "?"
                print(f"[{stage_name}] batch {batch_index}/{total_text}", flush=True)

    return torch.cat(image_batches), torch.cat(true_batches), torch.cat(pred_batches)


def _holdout_dataset(name: str, config: ExperimentConfig):
    root = str(config.resolved_data_dir())
    normalized = name.strip().lower()
    if normalized == "mnist_test":
        return datasets.MNIST(root=root, train=False, download=True, transform=build_eval_transform(config))
    if normalized == "emnist_digits_test":
        return datasets.EMNIST(
            root=root,
            split="digits",
            train=False,
            download=True,
            transform=build_eval_transform(config, correct_emnist=True),
        )
    if normalized == "qmnist_test10k":
        return datasets.QMNIST(
            root=root,
            what="test10k",
            compat=True,
            download=True,
            transform=build_eval_transform(config),
        )
    raise ValueError(f"不支持的 holdout 名称: {name}")


def evaluate_holdout(model, name: str, config: ExperimentConfig, output_dir: Path, device: str):
    if config.verbose:
        print(f"[holdout] start {name}", flush=True)
    dataset = _holdout_dataset(name, config)
    loader = DataLoader(
        dataset,
        batch_size=config.external_validation_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    if config.verbose:
        print(f"[holdout] {name} samples={len(dataset)} batches={len(loader)}", flush=True)
    images, y_true, y_pred = collect_predictions(model, loader, device=device, config=config, stage_name=f"holdout:{name}")
    summary = save_evaluation_bundle(
        images=images,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=output_dir / name,
        num_classes=config.num_classes,
    )
    summary["macro_f1"] = float(f1_score(y_true.numpy(), y_pred.numpy(), average="macro", zero_division=0))
    summary["name"] = name
    if config.verbose:
        print(f"[holdout] done {name} accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}", flush=True)
    return summary


def evaluate_external_holdouts(model, config: ExperimentConfig, output_dir: Path, device: str):
    output_dir = Path(output_dir)
    summaries = []
    for name in config.external_holdout_names:
        summaries.append(evaluate_holdout(model, name, config, output_dir, device))

    summary_path = output_dir / "external_holdouts_summary.json"
    csv_path = output_dir / "external_holdouts_summary.csv"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "accuracy", "macro_f1", "num_samples"])
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def _mnist_c_zip_path(config: ExperimentConfig) -> Path:
    if config.mnist_c_zip is not None:
        return Path(config.mnist_c_zip)
    return config.resolved_data_dir() / "mnist_c" / "mnist_c.zip"


def evaluate_mnist_c_zip(model, config: ExperimentConfig, output_dir: Path, device: str):
    zip_path = _mnist_c_zip_path(config)
    if not zip_path.exists():
        raise FileNotFoundError(f"未找到 MNIST-C zip: {zip_path}")

    rows = []
    model.eval()
    with zipfile.ZipFile(zip_path) as archive:
        corruptions = sorted(
            {
                parts[1]
                for item in archive.namelist()
                if (parts := item.split("/")) and len(parts) == 3 and parts[2] == "test_images.npy"
            }
        )
        for corruption_index, corruption in enumerate(corruptions, start=1):
            if config.verbose:
                print(f"[mnist-c] corruption {corruption_index}/{len(corruptions)}: {corruption}", flush=True)
            with archive.open(f"mnist_c/{corruption}/test_images.npy") as image_file:
                images_np = np.load(image_file)
            with archive.open(f"mnist_c/{corruption}/test_labels.npy") as label_file:
                labels_np = np.load(label_file)

            if images_np.ndim == 4 and images_np.shape[-1] == 1:
                images_np = np.squeeze(images_np, axis=-1)
            if images_np.ndim != 3:
                raise ValueError(f"MNIST-C 图像形状应为 [N, H, W] 或 [N, H, W, 1]，实际为 {images_np.shape}")
            images = torch.from_numpy(images_np).float().unsqueeze(1) / 255.0
            images = (images - 0.5) / 0.5
            labels = torch.from_numpy(labels_np).long()
            pred_batches = []
            with torch.no_grad():
                for batch_index, start in enumerate(range(0, len(images), config.external_validation_batch_size), start=1):
                    batch = images[start : start + config.external_validation_batch_size].to(device)
                    pred_batches.append(model(batch).argmax(dim=1).cpu())
                    total_batches = (len(images) + config.external_validation_batch_size - 1) // config.external_validation_batch_size
                    if config.verbose and (batch_index == 1 or batch_index % max(1, config.log_interval) == 0 or batch_index == total_batches):
                        print(f"[mnist-c:{corruption}] batch {batch_index}/{total_batches}", flush=True)
            predictions = torch.cat(pred_batches)
            accuracy = float(accuracy_score(labels.numpy(), predictions.numpy()))
            macro_f1 = float(f1_score(labels.numpy(), predictions.numpy(), average="macro", zero_division=0))
            rows.append(
                {
                    "corruption": corruption,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "num_samples": int(len(labels)),
                }
            )
            if config.verbose:
                print(f"[mnist-c] done {corruption} accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mnist_c_corruption_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "mnist_c_corruption_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corruption", "accuracy", "macro_f1", "num_samples"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


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
    parser.add_argument(
        "--holdouts",
        nargs="*",
        default=None,
        help="额外评估的测试集，例如 mnist_test emnist_digits_test qmnist_test10k",
    )
    parser.add_argument("--include-mnist-c", action="store_true", help="额外评估 data/mnist_c/mnist_c.zip")
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
        external_holdout_names=tuple(args.holdouts) if args.holdouts else ExperimentConfig.external_holdout_names,
    )
    paths = ensure_project_paths(config)
    _, val_loader = create_dataloaders(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model_from_checkpoint(args.checkpoint, config, device)

    images, y_true, y_pred = collect_predictions(model, val_loader, device=device)
    summary = save_evaluation_bundle(images, y_true, y_pred, paths.evaluation_dir / "validation", num_classes=config.num_classes)
    print(f"验证集评估完成。accuracy={summary['accuracy']:.4f}，结果已保存到 {paths.evaluation_dir / 'validation'}")

    if args.holdouts is not None:
        holdout_summaries = evaluate_external_holdouts(model, config, paths.evaluation_dir / "holdouts", device)
        for item in holdout_summaries:
            print(f"{item['name']}: accuracy={item['accuracy']:.4f}, macro_f1={item['macro_f1']:.4f}")

    if args.include_mnist_c:
        mnist_c_rows = evaluate_mnist_c_zip(model, config, paths.evaluation_dir / "mnist_c", device)
        mean_accuracy = sum(row["accuracy"] for row in mnist_c_rows) / len(mnist_c_rows)
        print(f"mnist_c: mean corruption accuracy={mean_accuracy:.4f}")


if __name__ == "__main__":
    main()
