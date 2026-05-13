from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    project_root: Path
    dataset_name: str = "mnist"
    data_dir: Path | None = None
    output_dir: Path | None = None
    run_name: str | None = None

    model_name: str = "small_cnn"
    batch_size: int = 64
    validation_split: float = 0.2
    validation_source: str = "train_split"
    image_size: int = 28
    num_classes: int = 10
    in_channels: int = 1
    learning_rate: float = 1e-3
    epochs: int = 5
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    dataloader_timeout: int = 0

    dropout: float = 0.25
    label_smoothing: float = 0.0
    optimizer_type: str = "Adam"
    scheduler_type: str = "none"
    weight_decay: float = 0.0
    use_amp: bool = False
    allow_tf32: bool = False
    use_early_stopping: bool = False
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 1e-4

    rotation_degrees: float = 8.0
    translate_ratio: float = 0.08
    scale_min: float = 1.0
    scale_max: float = 1.0
    shear_degrees: float = 0.0
    use_random_affine: bool = True
    use_gaussian_blur: bool = False

    use_mnist: bool = True
    use_emnist_digits: bool = False
    use_usps: bool = False
    use_qmnist: bool = False
    emnist_max_samples: int | None = None
    usps_max_samples: int | None = None
    qmnist_max_samples: int | None = None
    mnist_max_samples: int | None = None
    max_samples: int | None = None

    external_holdout_names: tuple[str, ...] = ("mnist_test", "emnist_digits_test", "qmnist_test10k")
    external_validation_batch_size: int = 512
    mnist_c_zip: Path | None = None

    use_tta: bool = False
    tta_n: int = 1
    auto_invert: bool = True

    def resolved_data_dir(self) -> Path:
        if self.data_dir is not None:
            return Path(self.data_dir)
        return Path(self.project_root) / "data"

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return Path(self.output_dir)
        if self.run_name:
            return Path(self.project_root) / "outputs_runs" / self.run_name
        return Path(self.project_root) / "outputs"

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        if payload["data_dir"] is None:
            payload["data_dir"] = str(self.resolved_data_dir())
        if payload["output_dir"] is None:
            payload["output_dir"] = str(self.resolved_output_dir())
        return payload


@dataclass
class ProjectPaths:
    outputs_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    figures_dir: Path
    predictions_dir: Path
    evaluation_dir: Path
    hpo_dir: Path


def ensure_project_paths(config: ExperimentConfig) -> ProjectPaths:
    outputs_dir = config.resolved_output_dir()
    checkpoints_dir = outputs_dir / "checkpoints"
    logs_dir = outputs_dir / "logs"
    figures_dir = outputs_dir / "figures"
    predictions_dir = outputs_dir / "predictions"
    evaluation_dir = outputs_dir / "evaluation"
    hpo_dir = outputs_dir / "hpo"

    for path in [
        outputs_dir,
        checkpoints_dir,
        logs_dir,
        figures_dir,
        predictions_dir,
        evaluation_dir,
        hpo_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        predictions_dir=predictions_dir,
        evaluation_dir=evaluation_dir,
        hpo_dir=hpo_dir,
    )
