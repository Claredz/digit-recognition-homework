from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

_logger = logging.getLogger(__name__)


class LoaderKind(str, Enum):
    IDX = "idx"
    IMAGE_FOLDER = "image_folder"
    MNIST_C = "mnist_c"
    TORCHVISION = "torchvision"


@dataclass
class DomainSpec:
    name: str
    loader_kind: LoaderKind
    path: Path
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class DomainRegistry:
    def __init__(self):
        self._specs: dict[str, DomainSpec] = {}

    def register(self, spec: DomainSpec):
        if spec.name in self._specs:
            _logger.warning("overwriting domain %s", spec.name)
        self._specs[spec.name] = spec

    def get(self, name: str) -> DomainSpec | None:
        return self._specs.get(name)

    def list_available(self) -> list[str]:
        return sorted(self._specs.keys())

    def remove(self, name: str):
        self._specs.pop(name, None)

    def __len__(self):
        return len(self._specs)

    def __iter__(self):
        return iter(self._specs.values())

    def __contains__(self, name: str):
        return name in self._specs

    @classmethod
    def from_project_root(cls, project_root: Path):
        registry = cls()
        data_dir = Path(project_root) / "data"

        candidates: list[tuple[str, str, LoaderKind]] = [
            ("mnist", "MNIST", LoaderKind.TORCHVISION),
            ("qmnist", "QMNIST", LoaderKind.TORCHVISION),
            ("emnist", "EMNIST", LoaderKind.TORCHVISION),
            ("mnist_c", "mnist_c", LoaderKind.MNIST_C),
            ("kannada", "kannada_mnist", LoaderKind.IMAGE_FOLDER),
            ("hasyv2", "hasyv2_digits", LoaderKind.IMAGE_FOLDER),
            ("chars74k", "chars74k_digits", LoaderKind.IMAGE_FOLDER),
            ("penbased", "penbased_rendered", LoaderKind.IMAGE_FOLDER),
            ("testa", "test_A_images.idx3-ubyte(1)", LoaderKind.IDX),
        ]
        for name, folder, kind in candidates:
            p = data_dir / folder
            if p.exists():
                registry.register(DomainSpec(name=name, loader_kind=kind, path=p, weight=1.0))

        return registry


class MultiDomainDataset(Dataset):
    def __init__(self, datasets: dict[str, Dataset]):
        self._datasets = datasets
        self._offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, ds_key in enumerate(datasets):
            n = len(datasets[ds_key])
            self._offsets[name] = (offset, offset + n)
            offset += n

    def __len__(self):
        return sum(hi - lo for _, (lo, hi) in self._offsets.items())

    def __getitem__(self, idx):
        for domain_name, (lo, hi) in self._offsets.items():
            if lo <= idx < hi:
                image, label = self._datasets[domain_name][idx - lo]
                return image, label, domain_name
        raise IndexError(f"index {idx} out of range")


class DomainBalancedSampler(Sampler):
    def __init__(
        self,
        offsets: dict[str, tuple[int, int]],
        batch_size: int,
        samples_per_domain_per_batch: int | str = "auto",
        n_batches: int = 1000,
        weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        self._offsets = offsets
        self._batch_size = batch_size
        self._n_batches = n_batches
        self._weights = weights or {}
        self._rng = np.random.RandomState(seed)
        self._domain_names = sorted(offsets.keys())

        if isinstance(samples_per_domain_per_batch, int):
            self._samples_per = samples_per_domain_per_batch
        else:
            n_domains = len(self._domain_names)
            self._samples_per = max(1, batch_size // max(1, n_domains))

    def __len__(self):
        return self._n_batches * self._batch_size

    def __iter__(self):
        per_domain = {name: self._samples_per for name in self._domain_names}
        for name, weight in self._weights.items():
            if name in per_domain:
                per_domain[name] = max(1, int(self._samples_per * weight))

        for _ in range(self._n_batches):
            batch_indices: list[int] = []
            for domain_name in self._domain_names:
                lo, hi = self._offsets[domain_name]
                n = per_domain[domain_name]
                if hi - lo >= n:
                    batch_indices.extend(self._rng.randint(lo, hi, size=n).tolist())
                else:
                    batch_indices.extend(self._rng.randint(lo, hi, size=n).tolist())
            self._rng.shuffle(batch_indices)
            yield from batch_indices[: self._batch_size]
