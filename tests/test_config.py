from pathlib import Path

from src.config import ExperimentConfig, ensure_project_paths


def test_ensure_project_paths_creates_expected_directories(tmp_path):
    config = ExperimentConfig(project_root=tmp_path)

    paths = ensure_project_paths(config)

    assert paths.outputs_dir == tmp_path / "outputs"
    assert paths.checkpoints_dir.is_dir()
    assert paths.logs_dir.is_dir()
    assert paths.figures_dir.is_dir()
    assert paths.predictions_dir.is_dir()
    assert paths.evaluation_dir.is_dir()
    assert paths.hpo_dir.is_dir()


def test_ensure_project_paths_uses_run_name_without_outputs_submission(tmp_path):
    config = ExperimentConfig(project_root=tmp_path, run_name="smoke_medium")

    paths = ensure_project_paths(config)

    assert paths.outputs_dir == tmp_path / "outputs_runs" / "smoke_medium"


def test_experiment_config_defaults_match_digit_task(tmp_path):
    config = ExperimentConfig(project_root=tmp_path)

    assert config.dataset_name == "mnist"
    assert config.image_size == 28
    assert config.num_classes == 10
    assert config.in_channels == 1
    assert config.batch_size == 64
    assert config.validation_split == 0.2
    assert config.model_name == "small_cnn"
