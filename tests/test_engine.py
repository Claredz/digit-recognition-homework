import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig, ensure_project_paths
from src.engine import build_optimizer, build_scheduler, fit
from src.model import SmallCNN


def test_fit_writes_checkpoint_history_and_curve(tmp_path):
    features = torch.rand(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(features, labels), batch_size=8, shuffle=False)

    config = ExperimentConfig(project_root=tmp_path, epochs=2, learning_rate=1e-3)
    paths = ensure_project_paths(config)
    model = SmallCNN()

    history = fit(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        paths=paths,
        device="cpu",
    )

    assert (paths.checkpoints_dir / "best_model.pt").exists()
    assert (paths.logs_dir / "history.json").exists()
    assert (paths.figures_dir / "training_curves.png").exists()

    payload = json.loads((paths.logs_dir / "history.json").read_text())
    checkpoint = torch.load(paths.checkpoints_dir / "best_model.pt", map_location="cpu")
    assert len(history["train_loss"]) == 2
    assert payload["best_val_accuracy"] >= 0.0
    assert checkpoint["model_name"] == "small_cnn"
    assert "history" in checkpoint


def test_build_optimizer_and_scheduler_support_submission_choices(tmp_path):
    config = ExperimentConfig(
        project_root=tmp_path,
        optimizer_type="AdamW",
        scheduler_type="CosineAnnealingLR",
        weight_decay=1e-4,
    )
    model = SmallCNN()

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    assert optimizer.__class__.__name__ == "AdamW"
    assert scheduler.__class__.__name__ == "CosineAnnealingLR"
