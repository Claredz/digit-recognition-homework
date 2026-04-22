import argparse
import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import ExperimentConfig, ensure_project_paths
from src.model import SmallCNN


class PredictionImageDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = 28):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            path
            for path in self.image_dir.glob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not self.image_paths:
            raise ValueError(f"在目录 {self.image_dir} 下没有找到可用于预测的图片")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image), image_path.name


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
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig(project_root=args.project_root.resolve(), image_size=args.image_size)
    paths = ensure_project_paths(config)

    dataset = PredictionImageDataset(args.image_dir, image_size=config.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SmallCNN(num_classes=config.num_classes, in_channels=config.in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for images, filenames in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            rows.extend(zip(filenames, predictions))

    output_path = paths.predictions_dir / "predictions.csv"
    write_predictions_csv(rows, output_path)
    print(f"预测结果已导出到 {output_path}")


if __name__ == "__main__":
    main()
