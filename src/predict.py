import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import ExperimentConfig, ensure_project_paths
from src.evaluate import load_model_from_checkpoint

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def auto_invert_grayscale(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    array = np.asarray(gray, dtype=np.uint8)
    if float(array.mean()) > 127.0:
        array = 255 - array
    return Image.fromarray(array, mode="L")


def crop_digit_foreground(image: Image.Image, threshold: int = 20) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    ys, xs = np.where(array > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return image.convert("L")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return image.crop((left, top, right, bottom))


def resize_and_center(image: Image.Image, image_size: int = 28, padding: int = 4) -> Image.Image:
    image = image.convert("L")
    max_digit_size = image_size - 2 * padding
    width, height = image.size
    if width == 0 or height == 0:
        return Image.new("L", (image_size, image_size), color=0)
    scale = min(max_digit_size / width, max_digit_size / height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", (image_size, image_size), color=0)
    offset = ((image_size - new_size[0]) // 2, (image_size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def preprocess_digit_image(image: Image.Image, image_size: int = 28, auto_invert: bool = True) -> Image.Image:
    gray = image.convert("L")
    if auto_invert:
        gray = auto_invert_grayscale(gray)
    cropped = crop_digit_foreground(gray)
    return resize_and_center(cropped, image_size=image_size)


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
