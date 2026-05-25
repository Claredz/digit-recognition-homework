from pathlib import Path

import pytest

from src.experiment_config import assert_safe_output_dir, load_experiment_config, resolve_experiment_output_dir


def test_testa_experiment_defaults_to_outputs_runs():
    project_root = Path.cwd()
    config = load_experiment_config(project_root / "experiments" / "testa_finetune_from_generalist.yaml")

    output_dir = resolve_experiment_output_dir(config, project_root)

    assert output_dir == project_root / "outputs_runs" / "testa_finetune_from_generalist"


def test_testa_output_guard_rejects_outputs_submission():
    project_root = Path.cwd()

    with pytest.raises(ValueError):
        assert_safe_output_dir(project_root / "outputs_submission" / "scratch", project_root)
