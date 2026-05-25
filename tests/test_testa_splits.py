import numpy as np

from src.testa_robust_train import kfold_indices, stratified_indices


def test_testa_kfold_splits_have_no_overlap_and_cover_validation_once():
    labels = np.repeat(np.arange(10), 20)
    n_splits = 5
    all_validation = []

    for fold_index in range(n_splits):
        train_indices, val_indices = kfold_indices(labels, n_splits=n_splits, fold_index=fold_index, seed=42)
        assert set(train_indices).isdisjoint(val_indices)
        assert len(train_indices) + len(val_indices) == len(labels)
        all_validation.extend(val_indices)

    assert sorted(all_validation) == list(range(len(labels)))


def test_testa_kfold_split_is_stratified_per_class():
    labels = np.repeat(np.arange(10), 20)

    _train_indices, val_indices = kfold_indices(labels, n_splits=5, fold_index=0, seed=42)
    val_labels = labels[val_indices]

    assert {label: int((val_labels == label).sum()) for label in range(10)} == {label: 4 for label in range(10)}


def test_stratified_indices_keep_each_class_in_train_and_validation():
    labels = np.repeat(np.arange(10), 10)

    train_indices, val_indices = stratified_indices(labels, train_ratio=0.7, seed=42)

    assert set(train_indices).isdisjoint(val_indices)
    assert set(labels[train_indices]) == set(range(10))
    assert set(labels[val_indices]) == set(range(10))
