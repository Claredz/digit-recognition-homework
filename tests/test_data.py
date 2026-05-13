import torch

from src.config import ExperimentConfig
from src.data import FolderDigitsDataset, build_eval_transform, build_train_transform, create_dataloaders


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


def test_train_and_eval_transforms_return_digit_tensors(tmp_path):
    config = ExperimentConfig(project_root=tmp_path, image_size=28)
    image = torch.zeros(1, 28, 28)

    train_tensor = build_train_transform(config)(image)
    eval_tensor = build_eval_transform(config)(image)

    assert train_tensor.shape == (1, 28, 28)
    assert eval_tensor.shape == (1, 28, 28)


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
