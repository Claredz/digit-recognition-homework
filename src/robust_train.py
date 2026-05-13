from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config import ExperimentConfig, ensure_project_paths
from src.engine import fit
from src.evaluate import load_model_from_checkpoint
from src.model import count_model_parameters
from src.robust_data import create_robust_finetune_dataloaders


def _default_clean_checkpoint(config: ExperimentConfig) -> Path:
    candidates = [
        config.resolved_output_dir() / "checkpoints" / "checkpoint_clean_best.pth",
        Path(config.project_root) / "outputs_submission" / "checkpoints" / "checkpoint_clean_best.pth",
        Path(config.project_root) / "outputs_submission" / "checkpoints" / "best_model_state_09974.pt",
        Path(config.project_root) / "outputs_submission" / "checkpoints" / "best_model_state.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def freeze_backbone(model: torch.nn.Module):
    if not hasattr(model, "features"):
        return
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if hasattr(model, "classifier"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True


def unfreeze_all(model: torch.nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = True


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_robust_finetune(config: ExperimentConfig):
    clean_checkpoint = _default_clean_checkpoint(config)
    if config.clean_checkpoint_path is not None:
        requested_checkpoint = Path(config.clean_checkpoint_path)
        if not requested_checkpoint.is_absolute():
            requested_checkpoint = Path(config.project_root) / requested_checkpoint
        if requested_checkpoint.exists():
            clean_checkpoint = requested_checkpoint
        elif requested_checkpoint.name != "checkpoint_clean_best.pth":
            raise FileNotFoundError(f"未找到 clean/base checkpoint，无法进行 robust fine-tuning: {requested_checkpoint}")
    if not clean_checkpoint.exists():
        raise FileNotFoundError(f"未找到 clean/base checkpoint，无法进行 robust fine-tuning: {clean_checkpoint}")

    paths = ensure_project_paths(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint_payload = load_model_from_checkpoint(clean_checkpoint, config, device)
    train_loader, val_loader, data_metadata = create_robust_finetune_dataloaders(config)

    original_lr = config.learning_rate
    original_epochs = config.epochs
    original_checkpoint_name = config.checkpoint_name
    config.learning_rate = config.fine_tune_lr
    config.epochs = config.fine_tune_epochs
    config.checkpoint_name = "robust_expert_best.pt"

    histories = {}
    if config.freeze_backbone_first and config.freeze_epochs > 0:
        freeze_backbone(model)
        config.epochs = config.freeze_epochs
        config.checkpoint_name = "robust_expert_head_warmup.pt"
        histories["freeze_head"] = fit(model, train_loader, val_loader, config=config, paths=paths, device=device)
        unfreeze_all(model)
        config.epochs = max(1, config.fine_tune_epochs - config.freeze_epochs)
        config.checkpoint_name = "robust_expert_best.pt"

    histories["full_finetune"] = fit(model, train_loader, val_loader, config=config, paths=paths, device=device)
    total_params, trainable_params = count_model_parameters(model)

    robust_history_path = paths.logs_dir / "robust_finetune_history.json"
    _write_json(robust_history_path, histories)

    manifest = {
        "model_role": "Teacher-Private-Style Robust Expert",
        "base_model_role": "Clean Expert / pretraining-base model",
        "initialized_from_clean_checkpoint": True,
        "clean_checkpoint_path": str(clean_checkpoint),
        "robust_checkpoint_path": str(paths.checkpoints_dir / "robust_expert_best.pt"),
        "model_name": config.model_name,
        "fine_tune_lr": config.fine_tune_lr,
        "fine_tune_epochs": config.fine_tune_epochs,
        "freeze_backbone_first": config.freeze_backbone_first,
        "freeze_epochs": config.freeze_epochs if config.freeze_backbone_first else 0,
        "ema_enabled": False,
        "tta_enabled_for_inference": config.use_tta,
        "robust_aug_strength": config.robust_aug_strength,
        "data_sources": data_metadata,
        "sampling": {
            "use_robust_sampler": config.use_robust_sampler,
            "mnist_family_weight": config.mnist_family_weight,
            "local_digits_weight": config.local_digits_weight,
            "hasyv2_weight": config.hasyv2_weight,
            "chars74k_weight": config.chars74k_weight,
            "penbased_weight": config.penbased_weight,
            "optical_weight": config.optical_weight,
        },
        "best_val_accuracy": histories["full_finetune"].get("best_val_accuracy"),
        "best_epoch": histories["full_finetune"].get("best_epoch"),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint_payload_keys": sorted(checkpoint_payload.keys()) if isinstance(checkpoint_payload, dict) else [],
    }
    _write_json(paths.logs_dir / "robust_finetune_manifest.json", manifest)

    config.learning_rate = original_lr
    config.epochs = original_epochs
    config.checkpoint_name = original_checkpoint_name

    print(f"Robust fine-tuning 完成。checkpoint: {paths.checkpoints_dir / 'robust_expert_best.pt'}")
    print(f"日志: {paths.logs_dir / 'robust_finetune_manifest.json'}")
    return histories
