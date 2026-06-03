import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import ExperimentConfig, ensure_project_paths
from src.evaluate import load_model_from_checkpoint
from src.preprocess import auto_invert_grayscale, crop_digit_foreground, preprocess_to_mnist_style_image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def resize_and_center(image: Image.Image, image_size: int = 28, padding: int = 4) -> Image.Image:
    body_size = max(1, image_size - 2 * padding)
    return preprocess_to_mnist_style_image(image, image_size=image_size, body_size=body_size, auto_invert=False)


def preprocess_digit_image(image: Image.Image, image_size: int = 28, auto_invert: bool = True) -> Image.Image:
    return preprocess_to_mnist_style_image(image, image_size=image_size, auto_invert=auto_invert)


def build_prediction_transform(config: ExperimentConfig):
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


class PredictionImageDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = 28, auto_invert: bool = True):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.auto_invert = auto_invert
        self.image_paths = sorted(
            path for path in self.image_dir.glob("*") if path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise ValueError(f"在目录 {self.image_dir} 下没有找到可用于预测的图片")

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            processed = preprocess_digit_image(
                image,
                image_size=self.image_size,
                auto_invert=self.auto_invert,
            )
        return self.transform(processed), image_path.name


def predict_probabilities_with_tta(model, images: torch.Tensor, config: ExperimentConfig, device: str) -> torch.Tensor:
    """Return class probabilities, averaging probabilities across TTA views."""
    model.eval()
    images = images.to(device)

    probabilities_sum = torch.softmax(model(images), dim=1)
    if not config.use_tta or config.tta_n <= 1:
        return probabilities_sum

    augment = transforms.RandomAffine(
        degrees=5,
        translate=(0.04, 0.04),
        scale=(0.96, 1.04),
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=-1.0,
    )
    for _ in range(config.tta_n - 1):
        augmented = torch.stack([augment(image.cpu()).to(device) for image in images])
        probabilities_sum = probabilities_sum + torch.softmax(model(augmented), dim=1)
    return probabilities_sum / config.tta_n


def predict_with_tta(model, images: torch.Tensor, config: ExperimentConfig, device: str):
    """Backward-compatible prediction helper.

    Plain inference returns raw model logits exactly as before. When TTA is enabled,
    probabilities are averaged across views and converted to log-probabilities so argmax
    behavior and downstream softmax calls remain compatible with the previous logits API.
    """
    if not config.use_tta or config.tta_n <= 1:
        model.eval()
        return model(images.to(device))

    probabilities = predict_probabilities_with_tta(model, images, config, device)
    return torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))


def write_predictions_csv(rows, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "prediction"])
        writer.writerows(rows)


def _safe_debug_stem(filename: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in Path(filename).stem)


def save_preprocess_debug_visualization(
    image_path: Path,
    output_dir: Path,
    prediction: int,
    confidence: float,
    image_size: int = 28,
    auto_invert: bool = True,
) -> Path:
    """Save a side-by-side original/processed preprocessing debug image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        original = ImageOps.contain(image.convert("L"), (112, 112), method=Image.Resampling.BILINEAR)
        processed = preprocess_digit_image(image, image_size=image_size, auto_invert=auto_invert)

    scale = 4
    processed_large = processed.resize((image_size * scale, image_size * scale), Image.Resampling.NEAREST)
    panel_width = 256
    panel_height = 160
    panel = Image.new("L", (panel_width, panel_height), color=255)
    panel.paste(original, (8 + (112 - original.width) // 2, 32 + (112 - original.height) // 2))
    panel.paste(processed_large, (136, 32))

    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), "original", fill=0)
    draw.text((136, 8), "processed", fill=0)
    draw.text((8, 146), f"{image_path.name} pred={prediction} conf={confidence:.4f}", fill=0)

    debug_path = output_dir / f"{_safe_debug_stem(image_path.name)}_pred{prediction}_conf{confidence:.4f}.png"
    panel.save(debug_path)
    return debug_path


def parse_args():
    parser = argparse.ArgumentParser(description="对无标签数字图片进行预测并导出结果")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="small_cnn")
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument("--no-auto-invert", action="store_true")
    parser.add_argument("--debug-preprocess", action="store_true")
    parser.add_argument("--debug-preprocess-samples", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(
        project_root=args.project_root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        use_tta=args.use_tta,
        tta_n=args.tta_n,
        auto_invert=not args.no_auto_invert,
        debug_preprocess=args.debug_preprocess,
        debug_preprocess_samples=args.debug_preprocess_samples,
    )
    paths = ensure_project_paths(config)

    dataset = PredictionImageDataset(
        args.image_dir,
        image_size=config.image_size,
        auto_invert=config.auto_invert,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model_from_checkpoint(args.checkpoint, config, device)

    rows = []
    debug_saved = 0
    debug_dir = paths.outputs_dir / "debug_preprocess"
    with torch.no_grad():
        for images, filenames in loader:
            probabilities = predict_probabilities_with_tta(model, images, config, device)
            confidence_values, prediction_values = probabilities.max(dim=1)
            predictions = prediction_values.cpu().tolist()
            confidences = confidence_values.cpu().tolist()
            rows.extend(zip(filenames, predictions))

            if config.debug_preprocess and debug_saved < config.debug_preprocess_samples:
                for filename, prediction, confidence in zip(filenames, predictions, confidences):
                    if debug_saved >= config.debug_preprocess_samples:
                        break
                    save_preprocess_debug_visualization(
                        dataset.image_dir / filename,
                        debug_dir,
                        prediction=int(prediction),
                        confidence=float(confidence),
                        image_size=config.image_size,
                        auto_invert=config.auto_invert,
                    )
                    debug_saved += 1

    output_path = paths.predictions_dir / "predictions.csv"
    write_predictions_csv(rows, output_path)
    print(f"预测结果已导出到 {output_path}")


if __name__ == "__main__":
    main()
