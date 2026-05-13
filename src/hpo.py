from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import torch

from src.config import ExperimentConfig, ensure_project_paths
from src.data import create_dataloaders
from src.engine import fit
from src.model import build_model
from src.train import set_seed


def _candidate_configs(base_config: ExperimentConfig):
    learning_rates = [3e-4, 6e-4, 9e-4]
    dropouts = [0.15, 0.22, 0.30]
    weight_decays = [1e-6, 1e-5, 1e-4]
    trial_id = 0
    for lr in learning_rates:
        for dropout in dropouts:
            for weight_decay in weight_decays:
                yield trial_id, {
                    "learning_rate": lr,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "rotation_degrees": 7.5,
                    "translate_ratio": 0.06,
                    "scale_min": 0.92,
                    "scale_max": 1.10,
                    "shear_degrees": 5.0,
                }
                trial_id += 1


def run_hpo(
    base_config: ExperimentConfig,
    n_trials: int = 12,
    trial_epochs: int = 5,
    trial_max_samples: int | None = 12000,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root_paths = ensure_project_paths(base_config)
    rows = []
    best_row = None

    for trial_index, params in _candidate_configs(base_config):
        if trial_index >= n_trials:
            break
        trial_output_dir = root_paths.hpo_dir / f"trial_{trial_index:03d}"
        trial_config = replace(
            base_config,
            output_dir=trial_output_dir,
            run_name=None,
            epochs=trial_epochs,
            max_samples=trial_max_samples,
            learning_rate=params["learning_rate"],
            dropout=params["dropout"],
            weight_decay=params["weight_decay"],
            rotation_degrees=params["rotation_degrees"],
            translate_ratio=params["translate_ratio"],
            scale_min=params["scale_min"],
            scale_max=params["scale_max"],
            shear_degrees=params["shear_degrees"],
            checkpoint_name="best_model.pt",
        )
        set_seed(trial_config.seed + trial_index)
        paths = ensure_project_paths(trial_config)
        train_loader, val_loader = create_dataloaders(trial_config)
        model = build_model(trial_config).to(device)
        history = fit(model, train_loader, val_loader, config=trial_config, paths=paths, device=device)
        row = {
            "trial": trial_index,
            "best_val_accuracy": history["best_val_accuracy"],
            "best_epoch": history["best_epoch"],
            "checkpoint": str(paths.checkpoints_dir / "best_model.pt"),
            **params,
        }
        rows.append(row)
        if best_row is None or row["best_val_accuracy"] > best_row["best_val_accuracy"]:
            best_row = row

    hpo_dir = root_paths.hpo_dir
    hpo_dir.mkdir(parents=True, exist_ok=True)
    (hpo_dir / "hpo_trials.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (hpo_dir / "hpo_trials.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (hpo_dir / "best_params.json").write_text(json.dumps(best_row or {}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"best": best_row, "rows": rows, "hpo_dir": str(hpo_dir)}
