from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, ensure_project_paths
from src.evaluate import load_model_from_checkpoint
from src.predict import PredictionImageDataset, predict_with_tta, write_predictions_csv
from src.validation_board import build_validation_board_loaders, score_validation_board


def parse_weight_grid(value: str | None):
    if not value:
        return (0.75, 0.70, 0.65, 0.60, 0.55, 0.50)
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def ensemble_probabilities(clean_logits: torch.Tensor, robust_logits: torch.Tensor, clean_weight: float):
    clean_prob = torch.softmax(clean_logits, dim=1)
    robust_prob = torch.softmax(robust_logits, dim=1)
    return clean_weight * clean_prob + (1.0 - clean_weight) * robust_prob


def predict_batch_pair(clean_model, robust_model, images: torch.Tensor, config: ExperimentConfig, device: str, clean_weight: float):
    clean_logits = predict_with_tta(clean_model, images, config, device)
    robust_logits = predict_with_tta(robust_model, images, config, device)
    ensemble_probs = ensemble_probabilities(clean_logits, robust_logits, clean_weight)
    return clean_logits, robust_logits, ensemble_probs


def predict_image_folder(clean_model, robust_model, loader, config: ExperimentConfig, device: str, clean_weight: float):
    clean_rows = []
    robust_rows = []
    ensemble_rows = []
    with torch.no_grad():
        for images, filenames in loader:
            clean_logits, robust_logits, ensemble_probs = predict_batch_pair(
                clean_model,
                robust_model,
                images,
                config,
                device,
                clean_weight,
            )
            clean_predictions = clean_logits.argmax(dim=1).cpu().tolist()
            robust_predictions = robust_logits.argmax(dim=1).cpu().tolist()
            ensemble_predictions = ensemble_probs.argmax(dim=1).cpu().tolist()
            clean_rows.extend(zip(filenames, clean_predictions))
            robust_rows.extend(zip(filenames, robust_predictions))
            ensemble_rows.extend(zip(filenames, ensemble_predictions))
    return clean_rows, robust_rows, ensemble_rows


def evaluate_ensemble_loader(clean_model, robust_model, loader, config: ExperimentConfig, device: str, clean_weight: float):
    clean_model.eval()
    robust_model.eval()
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            clean_logits = clean_model(images.to(device))
            robust_logits = robust_model(images.to(device))
            predictions = ensemble_probabilities(clean_logits, robust_logits, clean_weight).argmax(dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_examples += int(labels.numel())
    return {"accuracy": total_correct / total_examples if total_examples else 0.0, "num_samples": total_examples}


def search_ensemble_weight(clean_model, robust_model, config: ExperimentConfig, device: str, output_dir: Path):
    loaders = build_validation_board_loaders(config)
    rows = []
    best = None
    for weight in config.ensemble_weight_grid:
        split_results = {
            name: evaluate_ensemble_loader(clean_model, robust_model, loader, config, device, clean_weight=weight)
            for name, loader in loaders.items()
        }
        score = score_validation_board(split_results)
        row = {
            "clean_weight": weight,
            "composite_score": score["composite_score"],
            **{f"{name}_acc": result["accuracy"] for name, result in split_results.items()},
        }
        rows.append(row)
        if best is None or row["composite_score"] > best["composite_score"]:
            best = row

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
    parser.add_argument("--weight-grid", default="0.75,0.70,0.65,0.60,0.55,0.50")
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
