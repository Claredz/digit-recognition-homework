import json

import torch

from src.evaluate import save_evaluation_bundle


def test_save_evaluation_bundle_writes_summary_confusion_report_and_grid(tmp_path):
    images = torch.rand(4, 1, 28, 28)
    y_true = torch.tensor([0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 0, 1])

    summary = save_evaluation_bundle(
        images=images,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=tmp_path,
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "classification_report.json").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "misclassified_grid.png").exists()

    payload = json.loads((tmp_path / "summary.json").read_text())
    assert summary["accuracy"] == 0.5
    assert payload["accuracy"] == 0.5
