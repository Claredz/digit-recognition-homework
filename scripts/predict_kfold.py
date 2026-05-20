"""Generate submission CSV using K-Fold ensemble (raw + TTA8).

Supports two input modes:
- --image-dir <path>   : predict on a folder of images (used for the hidden test set)
- --idx-images <path> --idx-labels <path> : predict on TestA IDX (current demo)

Per design (see eval_kfold_multiview):
- raw view (no preprocess) + TTA8 + mean across all checkpoints achieved 0.7223
- Adding plan_A or preprocess view did not help. Defaults follow that conclusion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.evaluate import load_model_from_checkpoint
from src.testa_robust_train import IdxTestADataset

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class RawImageFolderDataset(Dataset):
    """Folder of images, no preprocess; resize + grayscale + tensor + normalize only."""

    def __init__(self, image_dir: Path, image_size: int = 28):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(p for p in self.image_dir.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
        if not self.image_paths:
            raise ValueError(f"目录 {self.image_dir} 下没有可用图片")
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("L")), path.name


class IdxPredictionDataset(Dataset):
    """Wrap IdxTestADataset to yield (tensor, row_id) instead of (tensor, label)."""

    def __init__(self, image_path: Path, label_path: Path | None = None):
        self.base = IdxTestADataset(image_path, label_path if label_path else image_path, preprocess=False) if label_path else None
        if self.base is None:
            raise ValueError("IDX prediction currently expects both image+label paths.")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index):
        tensor, _label = self.base[index]
        return tensor, f"{index:05d}.png"


@torch.no_grad()
def predict_probabilities(model, loader, device: str, tta_n: int):
    model.eval()
    tta_augment = transforms.RandomAffine(
        degrees=5, translate=(0.04, 0.04), scale=(0.96, 1.04),
        interpolation=transforms.InterpolationMode.BILINEAR, fill=-1.0,
    )
    all_probs = []
    all_names = []
    for images, names in loader:
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1)
        for _ in range(max(0, tta_n - 1)):
            augmented = torch.stack([tta_augment(image.cpu()) for image in images]).to(device)
            probs = probs + torch.softmax(model(augmented), dim=1)
        if tta_n > 1:
            probs = probs / tta_n
        all_probs.append(probs.cpu())
        all_names.extend(list(names))
    return torch.cat(all_probs), all_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True, help="path1 path2 ...")
    parser.add_argument("--image-dir", type=Path, default=None, help="Folder mode: predict over images in this dir")
    parser.add_argument("--idx-images", type=Path, default=None, help="IDX mode: path to test_A_images.idx3-ubyte")
    parser.add_argument("--idx-labels", type=Path, default=None, help="IDX mode: path to test_A_labels.idx1-ubyte (only used to fetch tensors; labels ignored for submission)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs_submission" / "predictions" / "submission_kfold.csv")
    parser.add_argument("--probabilities-output", type=Path, default=None, help="Optional .pt path to save raw probabilities")
    args = parser.parse_args()

    if (args.image_dir is None) == (args.idx_images is None):
        raise SystemExit("must pass exactly one of --image-dir or --idx-images")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} tta_n: {args.tta_n}", flush=True)

    if args.image_dir is not None:
        dataset = RawImageFolderDataset(args.image_dir, image_size=args.image_size)
        mode = "image_folder"
        src = str(args.image_dir)
    else:
        dataset = IdxPredictionDataset(args.idx_images, args.idx_labels)
        mode = "idx"
        src = str(args.idx_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"source: {src} mode: {mode} n={len(dataset)}", flush=True)

    config = ExperimentConfig(project_root=PROJECT_ROOT, model_name="medium_cnn", num_classes=10, in_channels=1, image_size=args.image_size)

    aggregate = None
    aggregate_names: list[str] = []
    per_model_acc_info = []
    for ckpt_path in args.checkpoints:
        path = Path(ckpt_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"\n[load] {path.name}", flush=True)
        t0 = time.perf_counter()
        model, _ = load_model_from_checkpoint(path, config, device)
        probs, names = predict_probabilities(model, loader, device, args.tta_n)
        elapsed = time.perf_counter() - t0
        if aggregate is None:
            aggregate = probs
            aggregate_names = names
        else:
            if names != aggregate_names:
                raise RuntimeError("file ordering mismatch between checkpoints — should not happen for the same dataset")
            aggregate = aggregate + probs
        per_model_acc_info.append({"checkpoint": str(path), "elapsed_sec": round(elapsed, 2)})
        print(f"[load] {path.name} done in {elapsed:.1f}s", flush=True)

    aggregate = aggregate / len(args.checkpoints)
    predictions = aggregate.argmax(dim=1).tolist()
    confidences = aggregate.max(dim=1).values.tolist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "prediction"])
        for name, pred in zip(aggregate_names, predictions):
            writer.writerow([name, pred])

    if args.probabilities_output is not None:
        args.probabilities_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"filenames": aggregate_names, "probabilities": aggregate}, args.probabilities_output)

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({
        "mode": mode,
        "source": src,
        "tta_n": args.tta_n,
        "n_samples": len(aggregate_names),
        "n_checkpoints": len(args.checkpoints),
        "checkpoints": per_model_acc_info,
        "output_csv": str(args.output),
        "view": "raw",
        "fusion": "probability_mean(raw + TTA averaged within each model, then mean across models)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwrote: {args.output}")
    print(f"manifest: {manifest_path}")
    print(f"mean top1 confidence: {sum(confidences)/len(confidences):.4f}")


if __name__ == "__main__":
    main()
