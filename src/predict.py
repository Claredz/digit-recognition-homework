import argparse
import csv
from pathlib import Path

from PIL import Image
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


def predict_with_tta(model, images: torch.Tensor, config: ExperimentConfig, device: str):
    model.eval()
    images = images.to(device)
    if not config.use_tta or config.tta_n <= 1:
        return model(images)

    augment = transforms.RandomAffine(degrees=5, translate=(0.04, 0.04), scale=(0.96, 1.04))
    logits_sum = model(images)
    for _ in range(config.tta_n - 1):
        augmented = torch.stack([augment(image.cpu()).to(device) for image in images])
        logits_sum = logits_sum + model(augmented)
    return logits_sum / config.tta_n


def write_predictions_csv(rows, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "prediction"])
        writer.writerows(rows)


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
    with torch.no_grad():
        for images, filenames in loader:
            logits = predict_with_tta(model, images, config, device)
            predictions = logits.argmax(dim=1).cpu().tolist()
            rows.extend(zip(filenames, predictions))

    output_path = paths.predictions_dir / "predictions.csv"
    write_predictions_csv(rows, output_path)
    print(f"预测结果已导出到 {output_path}")


if __name__ == "__main__":
    main()
