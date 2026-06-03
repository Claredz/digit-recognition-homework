from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, ensure_project_paths
from src.evaluate import load_model_from_checkpoint
from src.predict import PredictionImageDataset, predict_probabilities_with_tta, write_predictions_csv
from src.validation_board import build_validation_board_loaders, score_validation_board


DEFAULT_WEIGHT_GRID_STRING = "0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30"


def parse_weight_grid(value: str | None):
    if not value:
        return ExperimentConfig(project_root=Path(".")).ensemble_weight_grid
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def ensemble_probabilities(clean_probabilities: torch.Tensor, robust_probabilities: torch.Tensor, clean_weight: float):
    return clean_weight * clean_probabilities + (1.0 - clean_weight) * robust_probabilities


def _validate_probability_tensor(probabilities: torch.Tensor, name: str) -> None:
    if probabilities.ndim != 2:
        raise ValueError(f"{name} 必须是 [n_samples, n_classes]，实际 shape={tuple(probabilities.shape)}")
    if not torch.isfinite(probabilities).all():
        raise ValueError(f"{name} 包含非有限值")


def average_probabilities(probability_tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not probability_tensors:
        raise ValueError("至少需要一个 probability tensor")
    first_shape = tuple(probability_tensors[0].shape)
    for index, probabilities in enumerate(probability_tensors):
        _validate_probability_tensor(probabilities, f"probability_tensors[{index}]")
        if tuple(probabilities.shape) != first_shape:
            raise ValueError(f"probability tensor shape 不一致: {tuple(probabilities.shape)} != {first_shape}")
    return torch.stack([probabilities.float() for probabilities in probability_tensors], dim=0).mean(dim=0)


def weighted_probability_fusion(specialist_probabilities: torch.Tensor, generalist_probabilities: torch.Tensor, weight: float) -> torch.Tensor:
    _validate_probability_tensor(specialist_probabilities, "specialist_probabilities")
    _validate_probability_tensor(generalist_probabilities, "generalist_probabilities")
    if tuple(specialist_probabilities.shape) != tuple(generalist_probabilities.shape):
        raise ValueError(
            "specialist/generalist probability shape 不一致: "
            f"{tuple(specialist_probabilities.shape)} != {tuple(generalist_probabilities.shape)}"
        )
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight 必须在 [0, 1]，收到 {weight}")
    return weight * specialist_probabilities + (1.0 - weight) * generalist_probabilities


def search_specialist_generalist_weight(
    specialist_probabilities: torch.Tensor,
    generalist_probabilities: torch.Tensor,
    labels: torch.Tensor,
    weights: list[float] | tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> dict:
    labels = labels.cpu().long()
    rows = []
    best = None
    for weight in weights:
        fused = weighted_probability_fusion(specialist_probabilities, generalist_probabilities, float(weight))
        predictions = fused.argmax(dim=1).cpu()
        accuracy = float((predictions == labels).float().mean().item()) if labels.numel() else 0.0
        row = {"weight": float(weight), "accuracy": accuracy}
        rows.append(row)
        if best is None or accuracy > best["accuracy"]:
            best = row
    return {"best_weight": best["weight"] if best else None, "best_accuracy": best["accuracy"] if best else None, "rows": rows}


def predict_batch_pair(clean_model, robust_model, images: torch.Tensor, config: ExperimentConfig, device: str, clean_weight: float):
    clean_probabilities = predict_probabilities_with_tta(clean_model, images, config, device)
    robust_probabilities = predict_probabilities_with_tta(robust_model, images, config, device)
    ensemble_probs = ensemble_probabilities(clean_probabilities, robust_probabilities, clean_weight)
    return clean_probabilities, robust_probabilities, ensemble_probs


def predict_image_folder(clean_model, robust_model, loader, config: ExperimentConfig, device: str, clean_weight: float):
    clean_rows = []
    robust_rows = []
    ensemble_rows = []
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    with torch.no_grad():
        for batch_index, (images, filenames) in enumerate(loader, start=1):
            clean_probabilities, robust_probabilities, ensemble_probs = predict_batch_pair(
                clean_model,
                robust_model,
                images,
                config,
                device,
                clean_weight,
            )
            clean_predictions = clean_probabilities.argmax(dim=1).cpu().tolist()
            robust_predictions = robust_probabilities.argmax(dim=1).cpu().tolist()
            ensemble_predictions = ensemble_probs.argmax(dim=1).cpu().tolist()
            clean_rows.extend(zip(filenames, clean_predictions))
            robust_rows.extend(zip(filenames, robust_predictions))
            ensemble_rows.extend(zip(filenames, ensemble_predictions))
            if config.verbose and (batch_index == 1 or batch_index % max(1, config.log_interval) == 0 or batch_index == total_batches):
                total_text = str(total_batches) if total_batches is not None else "?"
                print(f"[predict] batch {batch_index}/{total_text}", flush=True)
    return clean_rows, robust_rows, ensemble_rows


def evaluate_ensemble_loader(clean_model, robust_model, loader, config: ExperimentConfig, device: str, clean_weight: float, split_name: str = "split"):
    clean_model.eval()
    robust_model.eval()
    total_correct = 0
    total_examples = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            labels = labels.to(device)
            images = images.to(device)
            clean_probabilities = torch.softmax(clean_model(images), dim=1)
            robust_probabilities = torch.softmax(robust_model(images), dim=1)
            predictions = ensemble_probabilities(clean_probabilities, robust_probabilities, clean_weight).argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_examples += int(labels.numel())
            if config.verbose and (batch_index == 1 or batch_index % max(1, config.log_interval) == 0 or batch_index == total_batches):
                total_text = str(total_batches) if total_batches is not None else "?"
                print(f"[ensemble:{clean_weight:.2f}:{split_name}] batch {batch_index}/{total_text}", flush=True)
    return {"accuracy": total_correct / total_examples if total_examples else 0.0, "num_samples": total_examples}


def search_ensemble_weight(clean_model, robust_model, config: ExperimentConfig, device: str, output_dir: Path):
    loaders = build_validation_board_loaders(config)
    rows = []
    best = None
    if config.verbose:
        print(f"[ensemble] searching clean weights: {config.ensemble_weight_grid}", flush=True)
    for weight in config.ensemble_weight_grid:
        if config.verbose:
            print(f"[ensemble] evaluate clean_weight={weight:.2f}", flush=True)
        split_results = {
            name: evaluate_ensemble_loader(clean_model, robust_model, loader, config, device, clean_weight=weight, split_name=name)
            for name, loader in loaders.items()
        }
        score = score_validation_board(split_results, config=config)
        row = {
            "clean_weight": weight,
            "composite_score": score["composite_score"],
            **{f"{name}_acc": result["accuracy"] for name, result in split_results.items()},
        }
        rows.append(row)
        if best is None or row["composite_score"] > best["composite_score"]:
            best = row
        if config.verbose:
            print(f"[ensemble] clean_weight={weight:.2f} composite={row['composite_score']:.4f} best={best['clean_weight']:.2f}/{best['composite_score']:.4f}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ensemble_weight_search.json"
    csv_path = output_dir / "ensemble_weight_search.csv"
    json_path.write_text(json.dumps({"rows": rows, "best": best}, indent=2, ensure_ascii=False), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return best, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Expert + Robust Expert 概率融合推理")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="medium_cnn")
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--robust-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-weight", type=float, default=0.60)
    parser.add_argument("--weight-grid", default=DEFAULT_WEIGHT_GRID_STRING)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument("--no-auto-invert", action="store_true")
    parser.add_argument("--search-validation-board", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        model_name=args.model_name,
        batch_size=args.batch_size,
        image_size=args.image_size,
        use_tta=args.use_tta,
        tta_n=args.tta_n,
        auto_invert=not args.no_auto_invert,
        ensemble_weight_clean=args.clean_weight,
        ensemble_weight_grid=parse_weight_grid(args.weight_grid),
    )
    paths = ensure_project_paths(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean_model, _ = load_model_from_checkpoint(args.clean_checkpoint, config, device)
    robust_model, _ = load_model_from_checkpoint(args.robust_checkpoint, config, device)

    chosen_weight = config.ensemble_weight_clean
    weight_search = None
    if args.search_validation_board:
        best, rows = search_ensemble_weight(clean_model, robust_model, config, device, paths.logs_dir)
        weight_search = {"best": best, "rows": rows}
        if best is not None:
            chosen_weight = float(best["clean_weight"])

    dataset = PredictionImageDataset(args.image_dir, image_size=config.image_size, auto_invert=config.auto_invert)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    clean_rows, robust_rows, ensemble_rows = predict_image_folder(
        clean_model,
        robust_model,
        loader,
        config,
        device,
        clean_weight=chosen_weight,
    )

    write_predictions_csv(clean_rows, paths.predictions_dir / "clean_predictions.csv")
    write_predictions_csv(robust_rows, paths.predictions_dir / "robust_predictions.csv")
    write_predictions_csv(ensemble_rows, paths.predictions_dir / "ensemble_predictions.csv")
    write_predictions_csv(ensemble_rows, paths.outputs_dir / "submission.csv")

    manifest = {
        "clean_checkpoint": str(args.clean_checkpoint),
        "robust_checkpoint": str(args.robust_checkpoint),
        "ensemble_formula": "final_prob = w * prob_clean + (1 - w) * prob_robust",
        "clean_weight": chosen_weight,
        "default_clean_weight": args.clean_weight,
        "weight_grid": list(config.ensemble_weight_grid),
        "used_validation_board_search": args.search_validation_board,
        "weight_search": weight_search,
        "use_tta": config.use_tta,
        "tta_n": config.tta_n,
        "outputs": {
            "clean_predictions": str(paths.predictions_dir / "clean_predictions.csv"),
            "robust_predictions": str(paths.predictions_dir / "robust_predictions.csv"),
            "ensemble_predictions": str(paths.predictions_dir / "ensemble_predictions.csv"),
            "submission": str(paths.outputs_dir / "submission.csv"),
        },
    }
    (paths.logs_dir / "ensemble_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Ensemble 推理完成。clean_weight={chosen_weight:.2f}")
    print(f"submission: {paths.outputs_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
