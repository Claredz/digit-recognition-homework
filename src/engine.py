import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.config import ExperimentConfig, ProjectPaths



def run_epoch(model, loader, criterion, device: str, optimizer=None):
    """运行一个 epoch，返回该 epoch 的平均 loss 与 accuracy。"""
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += images.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }



def save_history(history: dict, path: Path):
    """将训练/验证指标写入 JSON，便于复现实验与画图。"""
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")



def plot_history(history: dict, figure_path: Path):
    """将 loss/accuracy 曲线保存为 PNG 图片。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("损失")
    axes[0].set_xlabel("轮次")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title("准确率")
    axes[1].set_xlabel("轮次")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)



def fit(model, train_loader, val_loader, config: ExperimentConfig, paths: ProjectPaths, device: str):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {
        "config": config.to_dict(),
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_val_accuracy": 0.0,
    }

    best_val_accuracy = -1.0
    checkpoint_path = paths.checkpoints_dir / "best_model.pt"

    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device=device, optimizer=None)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

    history["best_val_accuracy"] = best_val_accuracy
    save_history(history, paths.logs_dir / "history.json")
    plot_history(history, paths.figures_dir / "training_curves.png")
    return history
