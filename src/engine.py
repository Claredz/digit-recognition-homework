import json
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.config import ExperimentConfig, ProjectPaths


def _configure_matplotlib_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def make_autocast_context(config: ExperimentConfig, device: str):
    if device == "cuda" and config.use_amp:
        return torch.amp.autocast("cuda")
    return nullcontext()


def make_grad_scaler(config: ExperimentConfig, device: str):
    return torch.amp.GradScaler("cuda", enabled=device == "cuda" and config.use_amp)


def run_epoch(
    model,
    loader,
    criterion,
    device: str,
    optimizer=None,
    config: ExperimentConfig | None = None,
    scaler=None,
    phase: str | None = None,
    epoch_index: int | None = None,
    total_epochs: int | None = None,
):
    training = optimizer is not None
    phase = phase or ("train" if training else "val")
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    verbose = True if config is None else config.verbose
    log_interval = max(1, 50 if config is None else config.log_interval)
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    start_time = time.perf_counter()

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        if device == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            autocast_context = make_autocast_context(config, device) if config is not None else nullcontext()
            with autocast_context:
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += images.size(0)

        should_log = verbose and (batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches)
        if should_log:
            running_loss = total_loss / max(1, total_examples)
            running_accuracy = total_correct / max(1, total_examples)
            epoch_text = ""
            if epoch_index is not None and total_epochs is not None:
                epoch_text = f" epoch {epoch_index}/{total_epochs}"
            batch_total_text = str(total_batches) if total_batches is not None else "?"
            lr_text = ""
            if optimizer is not None:
                lr_text = f" lr={optimizer.param_groups[0]['lr']:.6g}"
            print(
                f"[{phase}]{epoch_text} batch {batch_index}/{batch_total_text} "
                f"loss={running_loss:.4f} acc={running_accuracy:.4f}{lr_text} "
                f"elapsed={time.perf_counter() - start_time:.1f}s",
                flush=True,
            )

    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


def save_history(history: dict, path: Path):
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def plot_history(history: dict, figure_path: Path):
    _configure_matplotlib_chinese_font()
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


def build_optimizer(model, config: ExperimentConfig):
    optimizer_type = config.optimizer_type.strip().lower()
    if optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"不支持的 optimizer_type='{config.optimizer_type}'")


def build_scheduler(optimizer, config: ExperimentConfig):
    scheduler_type = config.scheduler_type.strip().lower()
    if scheduler_type in {"", "none", "null"}:
        return None
    if scheduler_type == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    if scheduler_type == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    raise ValueError(f"不支持的 scheduler_type='{config.scheduler_type}'")


def _scheduler_step(scheduler, config: ExperimentConfig, val_loss: float):
    if scheduler is None:
        return
    if config.scheduler_type.strip().lower() == "reducelronplateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def fit(model, train_loader, val_loader, config: ExperimentConfig, paths: ProjectPaths, device: str):
    if device == "cuda" and config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = model.to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = make_grad_scaler(config, device)

    history = {
        "config": config.to_dict(),
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
        "best_val_accuracy": 0.0,
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    bad_epochs = 0
    checkpoint_path = paths.checkpoints_dir / getattr(config, "checkpoint_name", "best_model.pt")

    print(
        f"Start training: model={config.model_name}, epochs={config.epochs}, "
        f"optimizer={config.optimizer_type}, scheduler={config.scheduler_type}, device={device}",
        flush=True,
    )

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            config=config,
            scaler=scaler,
            phase="train",
            epoch_index=epoch + 1,
            total_epochs=config.epochs,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            optimizer=None,
            config=config,
            phase="val",
            epoch_index=epoch + 1,
            total_epochs=config.epochs,
        )
        _scheduler_step(scheduler, config, val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["learning_rate"].append(current_lr)

        improved = val_metrics["accuracy"] > best_val_accuracy + config.early_stopping_min_delta
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            bad_epochs = 0
            history["best_val_accuracy"] = best_val_accuracy
            history["best_val_loss"] = best_val_loss
            history["best_epoch"] = epoch + 1
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "model_name": config.model_name,
                "epoch": epoch + 1,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "history": history,
            }
            torch.save(checkpoint, checkpoint_path)
        else:
            bad_epochs += 1

        elapsed = time.perf_counter() - epoch_start
        status = "best" if improved else f"no_improve={bad_epochs}"
        print(
            f"Epoch {epoch + 1:03d}/{config.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
            f"best_val_acc={best_val_accuracy:.4f}@{history['best_epoch']} | "
            f"lr={current_lr:.6g} | {status} | {elapsed:.1f}s",
            flush=True,
        )

        if config.use_early_stopping and bad_epochs >= config.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1}: best_val_acc={best_val_accuracy:.4f} "
                f"at epoch {history['best_epoch']}",
                flush=True,
            )
            break

    save_history(history, paths.logs_dir / "history.json")
    plot_history(history, paths.figures_dir / "training_curves.png")
    return history
