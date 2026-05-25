from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.ensemble_predict import average_probabilities, search_specialist_generalist_weight, weighted_probability_fusion
from src.evaluate import load_model_from_checkpoint
from src.experiment_config import (
    assert_safe_output_dir,
    config_list,
    config_section,
    load_experiment_config,
    resolve_experiment_output_dir,
    resolve_path,
)
from src.testa_robust_train import IdxTestADataset


DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "specialist_generalist_ensemble.yaml"
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class RawImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = 28):
        self.image_paths = sorted(path for path in Path(image_dir).glob("*") if path.suffix.lower() in _IMAGE_SUFFIXES)
        if not self.image_paths:
            raise ValueError(f"目录 {image_dir} 下没有可预测图片")
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("L")), path.name


class IdxPredictionDataset(Dataset):
    def __init__(self, image_path: Path, label_path: Path):
        self.base = IdxTestADataset(image_path, label_path, preprocess=False)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        tensor, _ = self.base[index]
        return tensor, f"{index:05d}.png"


@torch.no_grad()
def predict_probabilities(model, loader: DataLoader, device: str, tta_n: int):
    model.eval()
    tta_augment = transforms.RandomAffine(
        degrees=5,
        translate=(0.04, 0.04),
        scale=(0.96, 1.04),
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )
    probability_batches = []
    names = []
    for images, batch_names in loader:
        images = images.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images), dim=1)
        for _ in range(max(0, tta_n - 1)):
            augmented = torch.stack([tta_augment(image.cpu()) for image in images]).to(device)
            probabilities = probabilities + torch.softmax(model(augmented), dim=1)
        if tta_n > 1:
            probabilities = probabilities / tta_n
        probability_batches.append(probabilities.cpu())
        names.extend(list(batch_names))
    return torch.cat(probability_batches), names


@torch.no_grad()
def predict_labeled_probabilities(model, dataset: IdxTestADataset, batch_size: int, device: str):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    probabilities = []
    labels = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        probabilities.append(torch.softmax(model(images), dim=1).cpu())
        labels.append(batch_labels.cpu().long())
    return torch.cat(probabilities), torch.cat(labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Search/apply TestA specialist + generalist weighted fusion.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--idx-images", type=Path, default=None)
    parser.add_argument("--idx-labels", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-outputs-submission", action="store_true")
    return parser.parse_args()


def specialist_checkpoints(output_base: Path, raw_config: dict) -> list[Path]:
    seeds = [int(seed) for seed in config_list(raw_config, "seeds", [raw_config.get("seed", 42)])]
    n_splits = int(raw_config.get("folds", 5))
    checkpoint_name = str(config_section(raw_config, "ensemble").get("specialist_checkpoint_name", "testa_specialist_best.pt"))
    paths = []
    for seed in seeds:
        for fold_index in range(n_splits):
            path = output_base / f"seed_{seed}" / f"fold_{fold_index}" / "checkpoints" / checkpoint_name
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


def search_weight(raw_config: dict, project_root: Path, output_base: Path, device: str):
    model_config = config_section(raw_config, "model")
    prediction_config = config_section(raw_config, "prediction")
    ensemble_config = config_section(raw_config, "ensemble")
    generalist_checkpoint = resolve_path(model_config.get("generalist_checkpoint"), project_root)
    if generalist_checkpoint is None:
        raise ValueError("ensemble config 缺少 model.generalist_checkpoint")
    oof_path = output_base / "oof" / "oof_probabilities.pt"
    if not oof_path.exists():
        raise FileNotFoundError(f"未找到 specialist OOF 概率，请先运行 eval_testa_specialist_oof.py: {oof_path}")
    oof = torch.load(oof_path, map_location="cpu")
    sample_ids = [int(sample_id) for sample_id in oof["sample_ids"]]
    specialist_probs = oof["probabilities"].float()
    labels = oof["labels"].long()

    image_path = project_root / "data" / "test_A_images.idx3-ubyte(1)" / "test_A_images.idx3-ubyte"
    label_path = project_root / "data" / "test_A_labels.idx1-ubyte(1)" / "test_A_labels.idx1-ubyte"
    dataset = IdxTestADataset(image_path, label_path, preprocess=False, indices=sample_ids)
    exp_config = ExperimentConfig(
        project_root=project_root,
        model_name=str(model_config.get("model_name", "medium_cnn")),
        dropout=float(model_config.get("dropout", 0.21672530847241062)),
        batch_size=int(prediction_config.get("batch_size", 256)),
        image_size=28,
        verbose=False,
    )
    generalist_model, _ = load_model_from_checkpoint(generalist_checkpoint, exp_config, device)
    generalist_probs, generalist_labels = predict_labeled_probabilities(generalist_model, dataset, exp_config.batch_size, device)
    if not torch.equal(labels, generalist_labels):
        raise RuntimeError("specialist OOF labels 与 generalist OOF labels 顺序不一致")
    weights = [float(weight) for weight in ensemble_config.get("fusion_weights", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])]
    search = search_specialist_generalist_weight(specialist_probs, generalist_probs, labels, weights=weights)
    ensemble_dir = output_base / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment_id": raw_config["experiment_id"],
        "generalist_checkpoint": str(generalist_checkpoint),
        "oof_probabilities": str(oof_path),
        "weights": weights,
        "search": search,
    }
    (ensemble_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return search


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_config = load_experiment_config(args.config)
    output_base = resolve_experiment_output_dir(raw_config, project_root)
    assert_safe_output_dir(output_base, project_root, allow_outputs_submission=args.allow_outputs_submission)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    search = search_weight(raw_config, project_root, output_base, device)
    best_weight = float(search["best_weight"])
    print(json.dumps({"best_weight": best_weight, "best_accuracy": search["best_accuracy"]}, indent=2, ensure_ascii=False))
    if args.search_only:
        return

    if (args.image_dir is None) == (args.idx_images is None):
        raise SystemExit("final prediction requires exactly one of --image-dir or --idx-images, or pass --search-only")
    prediction_config = config_section(raw_config, "prediction")
    model_config = config_section(raw_config, "model")
    batch_size = int(prediction_config.get("batch_size", 256))
    tta_n = int(prediction_config.get("tta_n", 8))
    if args.image_dir is not None:
        dataset = RawImageFolderDataset(args.image_dir, image_size=28)
        source = str(args.image_dir)
    else:
        if args.idx_labels is None:
            raise SystemExit("IDX mode requires --idx-labels for tensor loading")
        dataset = IdxPredictionDataset(args.idx_images, args.idx_labels)
        source = str(args.idx_images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    exp_config = ExperimentConfig(
        project_root=project_root,
        model_name=str(model_config.get("model_name", "medium_cnn")),
        dropout=float(model_config.get("dropout", 0.21672530847241062)),
        batch_size=batch_size,
        image_size=28,
        verbose=False,
    )

    specialist_probs = []
    reference_names = None
    for checkpoint in specialist_checkpoints(output_base, raw_config):
        model, _ = load_model_from_checkpoint(checkpoint, exp_config, device)
        probabilities, names = predict_probabilities(model, loader, device, tta_n)
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise RuntimeError("specialist checkpoint prediction order mismatch")
        specialist_probs.append(probabilities)
    specialist_mean = average_probabilities(specialist_probs)

    generalist_checkpoint = resolve_path(model_config.get("generalist_checkpoint"), project_root)
    if generalist_checkpoint is None:
        raise ValueError("ensemble config 缺少 model.generalist_checkpoint")
    generalist_model, _ = load_model_from_checkpoint(generalist_checkpoint, exp_config, device)
    generalist_probs, names = predict_probabilities(generalist_model, loader, device, tta_n)
    if names != reference_names:
        raise RuntimeError("specialist/generalist prediction order mismatch")
    final_probs = weighted_probability_fusion(specialist_mean, generalist_probs, best_weight)
    predictions = final_probs.argmax(dim=1).tolist()

    output_path = args.output or (output_base / "ensemble" / "submission_fusion.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "prediction"])
        for name, prediction in zip(reference_names, predictions):
            writer.writerow([name, prediction])
    manifest = {
        "source": source,
        "output_csv": str(output_path),
        "best_weight": best_weight,
        "formula": "p_final = w * p_specialist_ensemble + (1 - w) * p_generalist",
        "tta_n": tta_n,
        "n_samples": len(reference_names),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"filenames": reference_names, "probabilities": final_probs}, output_path.with_suffix(".probabilities.pt"))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
