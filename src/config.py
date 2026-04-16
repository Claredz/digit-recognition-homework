from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
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
        if self.data_dir is not None:
            return self.data_dir
        return self.project_root / "data"

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        if payload["data_dir"] is None:
            payload["data_dir"] = str(self.resolved_data_dir())
        return payload


@dataclass
class ProjectPaths:
    outputs_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    figures_dir: Path
    predictions_dir: Path


def ensure_project_paths(config: ExperimentConfig) -> ProjectPaths:
    outputs_dir = config.project_root / "outputs"
    checkpoints_dir = outputs_dir / "checkpoints"
    logs_dir = outputs_dir / "logs"
    figures_dir = outputs_dir / "figures"
    predictions_dir = outputs_dir / "predictions"

    for path in [outputs_dir, checkpoints_dir, logs_dir, figures_dir, predictions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        predictions_dir=predictions_dir,
    )
