# Handwritten Digit Recognition Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a learning-oriented handwritten digit recognition baseline with a small CNN, configurable data loading, training/evaluation/prediction scripts, and saved experiment artifacts that are ready for later noise-oriented tuning.

**Architecture:** The codebase will stay small and explicit: one config module, one data module, one model module, one training engine, one training CLI, one evaluation CLI, and one prediction CLI. The first implementation validates the full pipeline on MNIST or a simple folder-based dataset, saves reproducible artifacts under `outputs/`, and keeps the data adapter isolated so the later teacher-provided noisy dataset can be integrated without rewriting the rest of the system.

**Tech Stack:** Python 3.11, PyTorch, torchvision, Pillow, scikit-learn, matplotlib, pytest

---

## File Map

### Files to create

- `requirements.txt` — runtime and test dependencies
- `.gitignore` — keep datasets, checkpoints, and caches out of git
- `README.md` — learner-oriented project overview and run commands
- `pytest.ini` — pytest discovery defaults
- `src/__init__.py` — package marker
- `src/config.py` — experiment config dataclass and output directory setup
- `src/data.py` — MNIST/folder dataset loading, transforms, and dataloaders
- `src/model.py` — small CNN baseline model
- `src/engine.py` — train/validate loop, checkpoint writing, history/curve outputs
- `src/train.py` — CLI entry point for training runs
- `src/evaluate.py` — CLI entry point for evaluation artifacts
- `src/predict.py` — checkpoint inference and CSV export for unlabeled images
- `tests/conftest.py` — shared test fixtures for temporary digit images
- `tests/test_config.py` — config and output path tests
- `tests/test_data.py` — dataset and dataloader tests
- `tests/test_model.py` — model forward/backward tests
- `tests/test_engine.py` — training loop artifact tests
- `tests/test_train_cli.py` — end-to-end training CLI smoke test
- `tests/test_evaluate.py` — confusion matrix and misclassification artifact tests
- `tests/test_predict.py` — prediction dataset and CSV export tests

### Responsibility boundaries

- `src/config.py` owns configuration only. It must not import torch or training code.
- `src/data.py` owns input data and transforms only. It must not know about optimization.
- `src/model.py` owns the CNN definition only.
- `src/engine.py` owns training mechanics and run artifacts only.
- `src/train.py`, `src/evaluate.py`, and `src/predict.py` are thin CLIs that glue modules together.
- Tests should stay file-local: each test file verifies one production module.

---

### Task 1: Bootstrap the project and runtime configuration

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `README.md`
- Create: `pytest.ini`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
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


def test_experiment_config_defaults_match_digit_task(tmp_path):
    config = ExperimentConfig(project_root=tmp_path)

    assert config.dataset_name == "mnist"
    assert config.image_size == 28
    assert config.num_classes == 10
    assert config.in_channels == 1
    assert config.batch_size == 64
    assert config.validation_split == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`.

- [ ] **Step 3: Write the minimal implementation and support files**

Create `requirements.txt`:

```text
torch>=2.3,<3.0
torchvision>=0.18,<1.0
numpy>=2.0,<3.0
pillow>=10.4,<12.0
matplotlib>=3.9,<4.0
scikit-learn>=1.5,<2.0
pytest>=8.3,<9.0
```

Create `.gitignore`:

```gitignore
__pycache__/
.pytest_cache/
.venv/
outputs/
data/
*.pyc
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
pythonpath = .
```

Create `src/__init__.py`:

```python
"""Handwritten digit recognition baseline package."""
```

Create `src/config.py`:

```python
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
```

Create `README.md`:

```markdown
# Handwritten Digit Recognition Homework

A learning-oriented handwritten digit recognition baseline for an AI introduction course.

## What this repository will do

- Train a small CNN on handwritten digits
- Save checkpoints, logs, and learning curves
- Generate confusion matrices and misclassification artifacts
- Export predictions for unlabeled test images later

## What you should learn from this project

1. How image data enters a classification pipeline
2. How a CNN turns images into class logits
3. How training, validation, and overfitting show up in metrics
4. How to read confusion matrices and misclassified samples
5. How to tune experiments systematically instead of editing code at random
```

- [ ] **Step 4: Install dependencies and rerun the targeted test**

Run:

```bash
python -m pip install -r requirements.txt
pytest tests/test_config.py -v
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore README.md pytest.ini src/__init__.py src/config.py tests/test_config.py
git commit -m "feat: bootstrap digit recognition project"
```

---

### Task 2: Build the dataset adapter and dataloaders

**Files:**
- Create: `src/data.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`:

```python
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def folder_digit_dataset(tmp_path):
    root = tmp_path / "digits"
    for label, pixel_value in [(0, 30), (1, 220)]:
        label_dir = root / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            image = Image.fromarray(
                np.full((20, 20), pixel_value + index, dtype=np.uint8),
                mode="L",
            )
            image.save(label_dir / f"{label}_{index}.png")
    return root
```

Create `tests/test_data.py`:

```python
import torch

from src.config import ExperimentConfig
from src.data import FolderDigitsDataset, create_dataloaders


def test_folder_digits_dataset_returns_grayscale_tensor(folder_digit_dataset, tmp_path):
    config = ExperimentConfig(
        project_root=tmp_path,
        dataset_name="folder",
        data_dir=folder_digit_dataset,
        image_size=28,
    )

    dataset = FolderDigitsDataset(folder_digit_dataset, image_size=config.image_size)
    image, label = dataset[0]

    assert image.shape == (1, 28, 28)
    assert image.dtype == torch.float32
    assert label in {0, 1}


def test_create_dataloaders_splits_dataset_deterministically(folder_digit_dataset, tmp_path):
    config = ExperimentConfig(
        project_root=tmp_path,
        dataset_name="folder",
        data_dir=folder_digit_dataset,
        batch_size=2,
        validation_split=0.25,
        seed=7,
    )

    train_loader, val_loader = create_dataloaders(config)

    train_count = len(train_loader.dataset)
    val_count = len(val_loader.dataset)

    assert train_count == 6
    assert val_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: Write the dataset implementation**

Create `src/data.py`:

```python
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from src.config import ExperimentConfig


class FolderDigitsDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 28, augment: bool = False):
        self.root = Path(root)
        self.transform = build_transform(image_size=image_size, augment=augment)
        self.samples: list[tuple[Path, int]] = []

        class_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        for class_dir in class_dirs:
            label = int(class_dir.name)
            for image_path in sorted(class_dir.glob("*")):
                if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"No labeled images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")
        tensor = self.transform(image)
        return tensor, label



def build_transform(image_size: int, augment: bool = False):
    operations: list = [transforms.Resize((image_size, image_size))]
    if augment:
        operations.extend(
            [
                transforms.RandomRotation(8),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return transforms.Compose(operations)



def load_base_dataset(config: ExperimentConfig):
    if config.dataset_name == "mnist":
        return datasets.MNIST(
            root=str(config.resolved_data_dir()),
            train=True,
            download=True,
            transform=build_transform(config.image_size, augment=False),
        )
    if config.dataset_name == "folder":
        return FolderDigitsDataset(
            root=config.resolved_data_dir(),
            image_size=config.image_size,
            augment=False,
        )
    raise ValueError(
        f"Unsupported dataset_name='{config.dataset_name}'. Use 'mnist' or 'folder'."
    )



def create_dataloaders(config: ExperimentConfig):
    dataset = load_base_dataset(config)

    val_size = max(1, int(len(dataset) * config.validation_split))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("validation_split leaves no training samples")

    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader
```

- [ ] **Step 4: Run tests to verify the data adapter passes**

Run:

```bash
pytest tests/test_data.py -v
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/conftest.py tests/test_data.py
git commit -m "feat: add dataset adapter and dataloaders"
```

---

### Task 3: Add the small CNN baseline model

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
import torch

from src.model import SmallCNN


def test_small_cnn_returns_logits_for_ten_classes():
    model = SmallCNN(num_classes=10, in_channels=1)
    batch = torch.randn(4, 1, 28, 28)

    logits = model(batch)

    assert logits.shape == (4, 10)



def test_small_cnn_backward_pass_populates_gradients():
    model = SmallCNN(num_classes=10, in_channels=1)
    batch = torch.randn(2, 1, 28, 28)
    labels = torch.tensor([0, 1])

    loss = torch.nn.CrossEntropyLoss()(model(batch), labels)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_model.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.model'`.

- [ ] **Step 3: Write the minimal CNN implementation**

Create `src/model.py`:

```python
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)
```

- [ ] **Step 4: Run tests to verify the model passes**

Run:

```bash
pytest tests/test_model.py -v
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "feat: add small cnn baseline"
```

---

### Task 4: Implement the training engine and artifact writing

**Files:**
- Create: `src/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_engine.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_engine.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.engine'`.

- [ ] **Step 3: Write the training engine**

Create `src/engine.py`:

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.config import ExperimentConfig, ProjectPaths



def run_epoch(model, loader, criterion, device: str, optimizer=None):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += images.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }



def save_history(history: dict, path: Path):
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")



def plot_history(history: dict, figure_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)



def fit(model, train_loader, val_loader, config: ExperimentConfig, paths: ProjectPaths, device: str):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {
        "config": config.to_dict(),
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "best_val_accuracy": 0.0,
    }

    best_val_accuracy = -1.0
    checkpoint_path = paths.checkpoints_dir / "best_model.pt"

    for epoch in range(config.epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device=device, optimizer=None)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "best_val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

    history["best_val_accuracy"] = best_val_accuracy
    save_history(history, paths.logs_dir / "history.json")
    plot_history(history, paths.figures_dir / "training_curves.png")
    return history
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
pytest tests/test_engine.py -v
```

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/engine.py tests/test_engine.py
git commit -m "feat: add training engine and artifacts"
```

---

### Task 5: Add the training CLI and run manifest

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train_cli.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_train_cli.py`:

```python
import json
import subprocess
import sys

import numpy as np
from PIL import Image



def write_digit(root, label, filename, pixel_value):
    label_dir = root / str(label)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.full((28, 28), pixel_value, dtype=np.uint8), mode="L")
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
```

- [ ] **Step 2: Run the integration test to verify it fails**

Run:

```bash
pytest tests/test_train_cli.py -v
```

Expected: FAIL with `No module named src.train`.

- [ ] **Step 3: Write the training CLI**

Create `src/train.py`:

```python
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.config import ExperimentConfig, ensure_project_paths
from src.data import create_dataloaders
from src.engine import fit
from src.model import SmallCNN



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def parse_args():
    parser = argparse.ArgumentParser(description="Train the handwritten digit CNN baseline")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-name", choices=["mnist", "folder"], default="mnist")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()



def main():
    args = parse_args()
    config = ExperimentConfig(
        project_root=args.project_root.resolve(),
        dataset_name=args.dataset_name,
        data_dir=args.data_dir.resolve() if args.data_dir is not None else None,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
    )

    set_seed(config.seed)
    paths = ensure_project_paths(config)
    train_loader, val_loader = create_dataloaders(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    history = fit(model, train_loader, val_loader, config=config, paths=paths, device=device)

    run_manifest = {
        "dataset_name": config.dataset_name,
        "data_dir": str(config.resolved_data_dir()),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "best_val_accuracy": history["best_val_accuracy"],
        "checkpoint": str(paths.checkpoints_dir / "best_model.pt"),
    }
    (paths.logs_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )

    print(
        f"Training complete. Best validation accuracy: {history['best_val_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run:

```bash
pytest tests/test_train_cli.py -v
```

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train_cli.py
git commit -m "feat: add training cli"
```

---

### Task 6: Add evaluation artifacts for confusion analysis

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
import json

import torch

from src.evaluate import save_evaluation_bundle



def test_save_evaluation_bundle_writes_summary_confusion_and_grid(tmp_path):
    images = torch.rand(4, 1, 28, 28)
    y_true = torch.tensor([0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 0, 1])

    save_evaluation_bundle(
        images=images,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=tmp_path,
    )

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "misclassified_grid.png").exists()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["accuracy"] == 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest tests/test_evaluate.py -v
```

Expected: FAIL with `No module named src.evaluate`.

- [ ] **Step 3: Write the evaluation module**

Create `src/evaluate.py`:

```python
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import ExperimentConfig, ensure_project_paths
from src.data import create_dataloaders
from src.model import SmallCNN



def save_evaluation_bundle(images: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = float(accuracy_score(y_true.numpy(), y_pred.numpy()))
    matrix = confusion_matrix(y_true.numpy(), y_pred.numpy(), labels=list(range(10)))

    (output_dir / "summary.json").write_text(
        json.dumps({"accuracy": accuracy}, indent=2),
        encoding="utf-8",
    )
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    wrong_indices = (y_true != y_pred).nonzero(as_tuple=False).flatten().tolist()
    if not wrong_indices:
        wrong_indices = [0]

    fig, axes = plt.subplots(1, min(4, len(wrong_indices)), figsize=(10, 3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for axis, index in zip(axes, wrong_indices[:4]):
        axis.imshow(images[index].squeeze(0).numpy(), cmap="gray")
        axis.set_title(f"T:{int(y_true[index])} P:{int(y_pred[index])}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "misclassified_grid.png")
    plt.close(fig)



def collect_predictions(model, loader, device: str):
    model.eval()
    image_batches = []
    true_batches = []
    pred_batches = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            image_batches.append(images.cpu())
            true_batches.append(labels.cpu())
            pred_batches.append(predictions)

    return torch.cat(image_batches), torch.cat(true_batches), torch.cat(pred_batches)



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained handwritten digit model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--dataset-name", choices=["mnist", "folder"], default="mnist")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=28)
    return parser.parse_args()



def main():
    args = parse_args()
    config = ExperimentConfig(
        project_root=args.project_root.resolve(),
        dataset_name=args.dataset_name,
        data_dir=args.data_dir.resolve() if args.data_dir is not None else None,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        image_size=args.image_size,
    )
    paths = ensure_project_paths(config)
    _, val_loader = create_dataloaders(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SmallCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    images, y_true, y_pred = collect_predictions(model, val_loader, device=device)
    save_evaluation_bundle(images, y_true, y_pred, paths.figures_dir)
    print(f"Evaluation complete. Accuracy saved to {paths.figures_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
pytest tests/test_evaluate.py -v
```

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation artifacts"
```

---

### Task 7: Add unlabeled-image prediction export

**Files:**
- Create: `src/predict.py`
- Test: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_predict.py`:

```python
import csv

import numpy as np
from PIL import Image

from src.predict import PredictionImageDataset, write_predictions_csv



def test_prediction_image_dataset_preserves_sorted_filenames(tmp_path):
    image_dir = tmp_path / "predict"
    image_dir.mkdir()

    for filename, pixel_value in [("b.png", 50), ("a.png", 150)]:
        Image.fromarray(np.full((28, 28), pixel_value, dtype=np.uint8), mode="L").save(
            image_dir / filename
        )

    dataset = PredictionImageDataset(image_dir, image_size=28)

    _, first_name = dataset[0]
    _, second_name = dataset[1]

    assert first_name == "a.png"
    assert second_name == "b.png"



def test_write_predictions_csv_outputs_expected_columns(tmp_path):
    output_path = tmp_path / "predictions.csv"

    write_predictions_csv(
        rows=[("sample_1.png", 7), ("sample_2.png", 3)],
        output_path=output_path,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"filename": "sample_1.png", "prediction": "7"},
        {"filename": "sample_2.png", "prediction": "3"},
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/test_predict.py -v
```

Expected: FAIL with `No module named src.predict`.

- [ ] **Step 3: Write the prediction module**

Create `src/predict.py`:

```python
import argparse
import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import ExperimentConfig, ensure_project_paths
from src.model import SmallCNN


class PredictionImageDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = 28):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            path
            for path in self.image_dir.glob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not self.image_paths:
            raise ValueError(f"No prediction images found under {self.image_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image), image_path.name



def write_predictions_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "prediction"])
        writer.writerows(rows)



def parse_args():
    parser = argparse.ArgumentParser(description="Predict labels for unlabeled digit images")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()



def main():
    args = parse_args()
    config = ExperimentConfig(project_root=args.project_root.resolve(), image_size=args.image_size)
    paths = ensure_project_paths(config)

    dataset = PredictionImageDataset(args.image_dir, image_size=config.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SmallCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for images, filenames in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            rows.extend(zip(filenames, predictions))

    output_path = paths.predictions_dir / "predictions.csv"
    write_predictions_csv(rows, output_path)
    print(f"Prediction export written to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_predict.py -v
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "feat: add prediction export"
```

---

## Manual Verification After Task 7

Run these commands from the repository root after all tasks are complete:

```bash
python -m pip install -r requirements.txt
pytest -q
python -m src.train --dataset-name mnist --project-root . --epochs 3 --batch-size 64
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pt --dataset-name mnist --project-root .
python -m src.predict --checkpoint outputs/checkpoints/best_model.pt --image-dir sample_predict_images --project-root .
```

Expected outcomes:
- `pytest -q` reports all tests passing.
- `outputs/checkpoints/best_model.pt` exists.
- `outputs/logs/history.json` and `outputs/logs/run_manifest.json` exist.
- `outputs/figures/training_curves.png`, `outputs/figures/confusion_matrix.png`, `outputs/figures/misclassified_grid.png` exist.
- `outputs/predictions/predictions.csv` exists.

## Learning Checkpoints

During implementation and review, make sure the engineer can explain these points out loud:

1. Why the data adapter is isolated in `src/data.py`.
2. Why `SmallCNN` outputs logits instead of probabilities.
3. Why validation accuracy matters more than training accuracy for model selection.
4. How confusion matrices reveal which digits are being confused.
5. Why experiment artifacts must be saved before the noisy teacher dataset arrives.

## Spec Coverage Map

- **Configurable data input path and dataset adapter** → Task 2
- **Small CNN baseline** → Task 3
- **Training/validation loop + best checkpoint** → Tasks 4 and 5
- **Accuracy reporting + experiment result logging** → Tasks 4 and 5
- **Confusion matrix + misclassification analysis** → Task 6
- **Batch prediction/export path for future test sets** → Task 7
- **Learning-oriented structure and documentation** → Task 1 + Learning Checkpoints section
