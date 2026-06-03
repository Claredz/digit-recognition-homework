"""Predict with the final fixed-weight TestA expert ensemble.

This is the submit-time entry point for the final 3-expert system:

    wide_resnet_tiny_raw_seed42       weight 0.7
    medium_v2_anti1_margin_seed2026  weight 0.2
    medium_v2_raw_seed3407           weight 0.1

Each expert is itself a 5-fold probability average. The script supports both
ordinary image folders and unlabeled IDX image files.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.evaluate import load_model_from_checkpoint

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

EXPERTS = [
    {
        "label": "wide_resnet_tiny_raw_seed42",
        "model_name": "wide_resnet_tiny",
        "weight": 0.7,
        "run": "testa_wide_resnet_tiny_raw_seed42_e60",
        "seed": 42,
    },
    {
        "label": "medium_anti1_seed2026",
        "model_name": "medium_cnn",
        "weight": 0.2,
        "run": "testa_medium_v2_anti1_margin_seed2026_e60",
        "seed": 2026,
    },
    {
        "label": "medium_raw_seed3407",
        "model_name": "medium_cnn",
        "weight": 0.1,
        "run": "testa_medium_v2_raw_seed3407_e60",
        "seed": 3407,
    },
]


class RawImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = 28):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            path for path in self.image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("L")), path.name


class UnlabeledIdxDataset(Dataset):
    def __init__(self, image_path: Path):
        self.image_path = Path(image_path)
        self.images = self._read_images(self.image_path)

    @staticmethod
    def _read_images(path: Path) -> torch.Tensor:
        payload = path.read_bytes()
        if len(payload) < 16:
            raise ValueError(f"IDX image file is too small: {path}")
        magic, count, rows, cols = struct.unpack(">IIII", payload[:16])
        if magic != 2051:
            raise ValueError(f"Not an IDX image file: {path}")
        raw = torch.frombuffer(payload, dtype=torch.uint8, offset=16)
        expected = count * rows * cols
        if raw.numel() != expected:
            raise ValueError(f"IDX image count mismatch: expected {expected}, got {raw.numel()}")
        return raw.reshape(count, rows, cols).clone()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index].float().unsqueeze(0) / 255.0
        image = TF.normalize(image, (0.5,), (0.5,))
        return image, f"{index:05d}.png"


def expert_fold_paths(project_root: Path, expert: dict) -> list[Path]:
    return [
        project_root
        / "outputs_runs"
        / expert["run"]
        / f"seed_{expert['seed']}"
        / f"fold_{fold}"
        / "checkpoints"
        / "testa_specialist_best.pt"
        for fold in range(5)
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--idx-images", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def predict_model_probabilities(model, loader: DataLoader, device: str, tta_n: int):
    model.eval()
    tta_augment = transforms.RandomAffine(
        degrees=5,
        translate=(0.04, 0.04),
        scale=(0.96, 1.04),
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )
    probability_batches = []
    names_out = []
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images), dim=1)
        for _ in range(max(0, tta_n - 1)):
            augmented = torch.stack([tta_augment(image.cpu()) for image in images]).to(device)
            probabilities = probabilities + torch.softmax(model(augmented), dim=1)
        if tta_n > 1:
            probabilities = probabilities / tta_n
        probability_batches.append(probabilities.cpu())
        names_out.extend(list(names))
    return torch.cat(probability_batches, dim=0), names_out


def build_dataset(args):
    if (args.image_dir is None) == (args.idx_images is None):
        raise SystemExit("Pass exactly one of --image-dir or --idx-images")
    if args.image_dir is not None:
        return RawImageFolderDataset(args.image_dir, image_size=args.image_size), str(args.image_dir)
    return UnlabeledIdxDataset(args.idx_images), str(args.idx_images)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    output_path = args.output or (
        project_root / "outputs_runs" / "final_weighted_ensemble_predictions" / "submission_final_weighted.csv"
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset, source = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} samples={len(dataset)} tta_n={args.tta_n}", flush=True)

    final_probabilities = None
    reference_names = None
    manifest_experts = []

    for expert in EXPERTS:
        fold_paths = expert_fold_paths(project_root, expert)
        missing = [path for path in fold_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints for {expert['label']}: {missing}")

        fold_sum = None
        exp_config = ExperimentConfig(
            project_root=project_root,
            model_name=expert["model_name"],
            batch_size=args.batch_size,
            image_size=args.image_size,
            verbose=False,
        )
        print(f"[expert] {expert['label']} weight={expert['weight']}", flush=True)
        for fold_index, checkpoint in enumerate(fold_paths):
            print(f"  [fold {fold_index}] {checkpoint}", flush=True)
            model, _ = load_model_from_checkpoint(checkpoint, exp_config, device)
            probabilities, names = predict_model_probabilities(model, loader, device, args.tta_n)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            if reference_names is None:
                reference_names = names
            elif names != reference_names:
                raise RuntimeError("Prediction order mismatch between folds")
            fold_sum = probabilities if fold_sum is None else fold_sum + probabilities

        expert_probabilities = fold_sum / len(fold_paths)
        weighted = float(expert["weight"]) * expert_probabilities
        final_probabilities = weighted if final_probabilities is None else final_probabilities + weighted
        manifest_experts.append(
            {
                "label": expert["label"],
                "model_name": expert["model_name"],
                "weight": expert["weight"],
                "checkpoints": [str(path) for path in fold_paths],
            }
        )

    predictions = final_probabilities.argmax(dim=1).tolist()
    confidences = final_probabilities.max(dim=1).values.tolist()
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "prediction"])
        for name, prediction in zip(reference_names, predictions):
            writer.writerow([name, prediction])

    probabilities_path = output_path.with_suffix(".probabilities.pt")
    torch.save({"filenames": reference_names, "probabilities": final_probabilities}, probabilities_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "source": source,
                "output_csv": str(output_path),
                "probabilities": str(probabilities_path),
                "n_samples": len(reference_names),
                "tta_n": args.tta_n,
                "batch_size": args.batch_size,
                "fusion": "sum_experts(weight * mean_fold_probabilities)",
                "experts": manifest_experts,
                "mean_top1_confidence": sum(confidences) / max(1, len(confidences)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote: {output_path}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"mean_top1_confidence={sum(confidences) / max(1, len(confidences)):.4f}", flush=True)


if __name__ == "__main__":
    main()
