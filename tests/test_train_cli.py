import json
import subprocess
import sys

import numpy as np
from PIL import Image


def write_digit(root, label, filename, pixel_value):
    label_dir = root / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)

    array = np.full((28, 28), pixel_value, dtype=np.uint8)
    image = Image.fromarray(array).convert("L")
    image.save(label_dir / filename)


def test_train_cli_runs_end_to_end_on_folder_dataset(tmp_path):
    data_root = tmp_path / "digits"
    for index in range(4):
        write_digit(data_root, 0, f"zero_{index}.png", 20 + index)
        write_digit(data_root, 1, f"one_{index}.png", 220 - index)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--dataset-name",
            "folder",
            "--data-dir",
            str(data_root),
            "--project-root",
            str(tmp_path),
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "outputs" / "checkpoints" / "best_model.pt").exists()
    assert (tmp_path / "outputs" / "logs" / "run_manifest.json").exists()

    manifest = json.loads((tmp_path / "outputs" / "logs" / "run_manifest.json").read_text())
    assert manifest["dataset_name"] == "folder"
    assert manifest["epochs"] == 1
