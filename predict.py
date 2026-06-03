"""predict.py — 手写数字识别提交脚本

系统运行命令:
    python3 predict.py --testdata /testdata --output /results/submission.csv

测试数据: MNIST IDX 格式 (test_B_images.idx3-ubyte)
输出格式: 每行一个数字，无表头

集成系统: 3 专家固定权重 (0.7 / 0.2 / 0.1)，每专家 5 折概率平均
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ---------------------------------------------------------------------------
# 模型定义 — 精确复制自 src/model.py 和 src/models/heterogeneous.py
# ---------------------------------------------------------------------------

class MediumCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class LargeCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


class PreActBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(inputs))
        shortcut = self.shortcut(out if not isinstance(self.shortcut, nn.Identity) else inputs)
        out = self.conv1(out)
        out = self.conv2(self.drop(self.act2(self.bn2(out))))
        return out + shortcut


class PreActResNetTiny(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout: float = 0.1, widths: tuple = (32, 64, 128)):
        super().__init__()
        c1, c2, c3 = widths
        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False)
        self.stage1 = nn.Sequential(PreActBlock(c1, c1, dropout=dropout), PreActBlock(c1, c1, dropout=dropout))
        self.stage2 = nn.Sequential(PreActBlock(c1, c2, stride=2, dropout=dropout), PreActBlock(c2, c2, dropout=dropout))
        self.stage3 = nn.Sequential(PreActBlock(c2, c3, stride=2, dropout=dropout), PreActBlock(c3, c3, dropout=dropout))
        self.head = nn.Sequential(nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c3, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.stem(inputs)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return self.head(out)


# ---------------------------------------------------------------------------
# IDX 图像加载
# ---------------------------------------------------------------------------

def load_idx_images(filepath: Path) -> torch.Tensor:
    """读取 MNIST IDX3-UBYTE 图像，返回 (N, 1, 28, 28) float32 归一化到 [-1, 1]"""
    payload = filepath.read_bytes()
    if len(payload) < 16:
        raise ValueError(f"IDX 文件太小: {filepath}")
    magic, count, rows, cols = struct.unpack(">IIII", payload[:16])
    if magic != 2051:
        raise ValueError(f"不是 IDX 图像文件 (magic={magic}): {filepath}")
    raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8, offset=16)
    expected = count * rows * cols
    if raw.numel() != expected:
        raise ValueError(f"IDX 图像数量不匹配: 预期 {expected}, 实际 {raw.numel()}")
    images = raw.reshape(count, 1, rows, cols).float() / 255.0
    images = (images - 0.5) / 0.5
    return images


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(checkpoint_path: Path, device: str) -> nn.Module:
    """从 checkpoint 加载模型，根据 model_name 分派到正确架构"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "")
    dropout = checkpoint.get("config", {}).get("dropout", 0.1)

    if model_name == "wide_resnet_tiny":
        model = PreActResNetTiny(num_classes=10, in_channels=1, dropout=dropout, widths=(48, 96, 192))
    elif model_name == "preact_resnet_tiny":
        model = PreActResNetTiny(num_classes=10, in_channels=1, dropout=dropout, widths=(32, 64, 128))
    elif model_name == "medium_cnn":
        model = MediumCNN(num_classes=10, in_channels=1, dropout=dropout)
    elif model_name == "large_cnn":
        model = LargeCNN(num_classes=10, in_channels=1, dropout=dropout)
    else:
        raise ValueError(f"未知的 model_name={model_name!r}，checkpoint: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_tta(model: nn.Module, images: torch.Tensor, device: str, tta_n: int) -> torch.Tensor:
    """返回 (N, 10) softmax 概率，支持 TTA"""
    model.eval()
    images = images.to(device)
    probs = torch.softmax(model(images), dim=1)

    if tta_n > 1:
        augment = transforms.RandomAffine(
            degrees=5,
            translate=(0.04, 0.04),
            scale=(0.96, 1.04),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=-1.0,
        )
        for _ in range(tta_n - 1):
            augmented = torch.stack([augment(img.cpu()) for img in images]).to(device)
            probs = probs + torch.softmax(model(augmented), dim=1)
        probs = probs / tta_n

    return probs.cpu()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

EXPERTS = [
    {"name": "wide_resnet_tiny_raw_seed42", "model_name": "wide_resnet_tiny", "weight": 0.7},
    {"name": "medium_anti1_seed2026",       "model_name": "medium_cnn",        "weight": 0.2},
    {"name": "medium_raw_seed3407",         "model_name": "medium_cnn",        "weight": 0.1},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手写数字识别预测")
    parser.add_argument("--testdata", required=True, help="测试数据目录，包含 test_B_images.idx3-ubyte")
    parser.add_argument("--output", required=True, help="预测结果输出路径 (submission.csv)")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--tta-n", type=int, default=8, help="TTA 视图数 (0 或 1 禁用)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # 加载测试图像
    testdata_dir = Path(args.testdata)
    idx_path = testdata_dir / "test_B_images.idx3-ubyte"
    if not idx_path.exists():
        print(f"错误：找不到测试图像文件 {idx_path}", file=sys.stderr)
        sys.exit(1)

    images = load_idx_images(idx_path)
    n_samples = images.shape[0]
    print(f"测试集: {n_samples} 样本, 设备: {device}, TTA: {args.tta_n}", flush=True)

    # predict.py 所在目录
    root = Path(__file__).resolve().parent

    # 加权集成
    final_probs = None

    for expert in EXPERTS:
        fold_probs = None
        print(f"[专家] {expert['name']} 权重={expert['weight']}", flush=True)

        for fold in range(5):
            ckpt_path = root / "models" / expert["name"] / f"fold_{fold}" / "testa_specialist_best.pt"
            if not ckpt_path.exists():
                print(f"错误：找不到 checkpoint {ckpt_path}", file=sys.stderr)
                sys.exit(1)

            print(f"  [fold {fold}] 加载 {ckpt_path}", flush=True)
            model = build_model_from_checkpoint(ckpt_path, device)

            # 分 batch 推理
            batch_probs_list = []
            for start in range(0, n_samples, args.batch_size):
                end = min(start + args.batch_size, n_samples)
                batch_images = images[start:end]
                batch_probs = predict_with_tta(model, batch_images, device, args.tta_n)
                batch_probs_list.append(batch_probs)

            fold_result = torch.cat(batch_probs_list, dim=0)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            fold_probs = fold_result if fold_probs is None else fold_probs + fold_result

        expert_probs = fold_probs / 5.0
        weighted = expert["weight"] * expert_probs
        final_probs = weighted if final_probs is None else final_probs + weighted

    predictions = final_probs.argmax(dim=1).tolist()
    confidences = final_probs.max(dim=1).values.tolist()

    # 写输出：每行一个数字，无表头
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(f"{p}\n")

    mean_conf = sum(confidences) / max(1, len(confidences))
    print(f"预测完成: {output_path}, 平均置信度={mean_conf:.4f}", flush=True)


if __name__ == "__main__":
    main()
