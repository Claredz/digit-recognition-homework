import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig, ensure_project_paths
from src.engine import fit
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
    assert len(history["train_loss"]) == 2
    assert payload["best_val_accuracy"] >= 0.0
