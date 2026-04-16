from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration for the handwritten digit recognition task."""

    project_root: Path

    dataset_name: str = "mnist"
    data_dir: Path | None = None

    batch_size: int = 64
    validation_split: float = 0.2

    image_size: int = 28
    num_classes: int = 10
    in_channels: int = 1

    learning_rate: float = 1e-3
    epochs: int = 5
    seed: int = 42
    num_workers: int = 0

    def resolved_data_dir(self) -> Path:
        """Return the directory where raw datasets should be stored."""

        if self.data_dir is not None:
            return self.data_dir
        return self.project_root / "data"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a JSON-friendly dictionary."""

        resolved_data_dir = self.resolved_data_dir()

        return {
            "project_root": str(self.project_root),
            "dataset_name": self.dataset_name,
            "data_dir": str(resolved_data_dir),
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "seed": self.seed,
            "num_workers": self.num_workers,
        }


@dataclass(frozen=True)
class ProjectPaths:
    outputs_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    figures_dir: Path
    predictions_dir: Path


def ensure_project_paths(config: ExperimentConfig) -> ProjectPaths:
    """Create and return the standard output directories for an experiment."""

    outputs_dir = config.project_root / "outputs"
    checkpoints_dir = outputs_dir / "checkpoints"
    logs_dir = outputs_dir / "logs"
    figures_dir = outputs_dir / "figures"
    predictions_dir = outputs_dir / "predictions"

    for p in (outputs_dir, checkpoints_dir, logs_dir, figures_dir, predictions_dir):
        p.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        predictions_dir=predictions_dir,
    )
