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


# ---------------------------------------------------------------------------
# 5专家 Dynamic MoE 配置
# ---------------------------------------------------------------------------

EXPERTS = [
    {
        "name": "wide_resnet_tiny_raw_seed42",
        "kind": "kfold",
        "model_name": "wide_resnet_tiny",
        "base_weight": 0.30,
        "static_weight": 0.40,
        "anti1": 0.0,
    },
    {
        "name": "medium_anti1_seed2026",
        "kind": "kfold",
        "model_name": "medium_cnn",
        "base_weight": 0.15,
        "static_weight": 0.10,
        "anti1": 1.0,
    },
    {
        "name": "medium_raw_seed3407",
        "kind": "kfold",
        "model_name": "medium_cnn",
        "base_weight": 0.15,
        "static_weight": 0.30,
        "anti1": 0.0,
    },
    {
        "name": "robust_v1",
        "kind": "single",
        "checkpoint": "robust_expert_best.pt",
        "model_name": "medium_cnn",
        "base_weight": 0.30,
        "static_weight": 0.15,
        "anti1": 0.0,
    },
    {
        "name": "MNIST_clean",
        "kind": "single",
        "checkpoint": "best_model_state.pt",
        "model_name": "medium_cnn",
        "base_weight": 0.10,
        "static_weight": 0.05,
        "anti1": 0.0,
    },
]

DYNAMIC_PARAMS = {
    "confidence": 2.5,
    "margin": 1.0,
    "disagreement": 0.0,
    "anti1_boost": 0.25,
    "clean_conf_boost": 1.0,
    "robust_conf_boost": 1.0,
}

RULE_TEMPLATES = {
    "testa":    [0.55, 0.15, 0.20, 0.08, 0.02],
    "mnist":    [0.20, 0.10, 0.25, 0.15, 0.30],
    "robust":   [0.25, 0.10, 0.20, 0.40, 0.05],
    "anti1":    [0.35, 0.35, 0.15, 0.10, 0.05],
    "balanced": [0.30, 0.15, 0.25, 0.20, 0.10],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5专家动态 MoE 手写数字识别预测")
    parser.add_argument("--testdata", required=True, help="测试数据目录，包含 test_B_images.idx3-ubyte")
    parser.add_argument("--output", required=True, help="预测结果输出路径")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tta-n", type=int, default=8)
    parser.add_argument("--router", choices=["dynamic", "rule", "static", "average"], default="dynamic")
    return parser.parse_args()


def load_expert_model(root: Path, expert: dict, fold: int | None, device: str):
    if expert["kind"] == "kfold":
        ckpt_path = root / "models" / expert["name"] / f"fold_{fold}" / "testa_specialist_best.pt"
    else:
        ckpt_path = root / "models" / expert["name"] / expert["checkpoint"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    return build_model_from_checkpoint(ckpt_path, device)


@torch.no_grad()
def predict_expert_probabilities(root: Path, expert: dict, images: torch.Tensor, device: str, batch_size: int, tta_n: int) -> torch.Tensor:
    n = images.shape[0]
    if expert["kind"] == "kfold":
        fold_sum = None
        for fold in range(5):
            model = load_expert_model(root, expert, fold, device)
            batches = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batches.append(predict_with_tta(model, images[start:end], device, tta_n))
            fold_probs = torch.cat(batches, dim=0)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            fold_sum = fold_probs if fold_sum is None else fold_sum + fold_probs
        return fold_sum / 5.0

    model = load_expert_model(root, expert, None, device)
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batches.append(predict_with_tta(model, images[start:end], device, tta_n))
    probs = torch.cat(batches, dim=0)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return probs


def compute_dynamic_weights(probs: torch.Tensor) -> torch.Tensor:
    base = torch.tensor([expert["base_weight"] for expert in EXPERTS], dtype=probs.dtype, device=probs.device)
    base = base / base.sum()
    anti1_vector = torch.tensor([expert["anti1"] for expert in EXPERTS], dtype=probs.dtype, device=probs.device)

    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]
    mean_distribution = (probs * base.view(1, -1, 1)).sum(dim=1, keepdim=True).clamp_min(1e-9)
    disagreement = (probs.clamp_min(1e-9) * torch.log(probs.clamp_min(1e-9) / mean_distribution)).sum(dim=-1)

    confidence_c = confidence - confidence.mean(dim=1, keepdim=True)
    margin_c = margin - margin.mean(dim=1, keepdim=True)
    disagreement_c = disagreement - disagreement.mean(dim=1, keepdim=True)

    fixed_probs = (probs * base.view(1, -1, 1)).sum(dim=1)
    fixed_pred = fixed_probs.argmax(dim=1)
    anti1_signal = (fixed_pred == 1).float().view(-1, 1) * anti1_vector.view(1, -1)

    clean_signal = torch.zeros_like(confidence)
    clean_signal[:, 4] = confidence[:, 4] - confidence.mean(dim=1)
    robust_signal = torch.zeros_like(confidence)
    robust_signal[:, 3] = confidence[:, 3] - confidence.mean(dim=1)

    p = DYNAMIC_PARAMS
    score = (
        torch.log(base).view(1, -1)
        + p["confidence"] * confidence_c
        + p["margin"] * margin_c
        - p["disagreement"] * disagreement_c
        + p["anti1_boost"] * anti1_signal
        + p["clean_conf_boost"] * clean_signal
        + p["robust_conf_boost"] * robust_signal
    )
    return torch.softmax(score, dim=-1)


def compute_rule_weights(probs: torch.Tensor) -> torch.Tensor:
    templates = {key: torch.tensor(value, dtype=probs.dtype, device=probs.device) for key, value in RULE_TEMPLATES.items()}
    max_probs, preds = probs.max(dim=-1)
    wide_conf = max_probs[:, 0]
    medium_conf = (max_probs[:, 1] + max_probs[:, 2]) * 0.5
    robust_conf = max_probs[:, 3]
    clean_conf = max_probs[:, 4]
    medium_agree = preds[:, 1] == preds[:, 2]
    clean_agrees = preds[:, 4] == preds[:, 2]
    robust_high = robust_conf >= torch.maximum(wide_conf, medium_conf) - 0.03
    mnist_like = (clean_conf >= 0.82) & clean_agrees
    testa_like = (wide_conf >= 0.78) & ((preds[:, 0] == preds[:, 1]) | (preds[:, 0] == preds[:, 2]))
    anti1_risk = ((probs[:, :, 1].max(dim=1).values >= 0.45) & (preds[:, 1] != 1) & (preds[:, 0] == 1))

    weights = templates["balanced"].view(1, -1).repeat(probs.shape[0], 1)
    weights = torch.where(testa_like.view(-1, 1), templates["testa"].view(1, -1), weights)
    weights = torch.where(mnist_like.view(-1, 1), templates["mnist"].view(1, -1), weights)
    weights = torch.where(robust_high.view(-1, 1), templates["robust"].view(1, -1), weights)
    weights = torch.where(anti1_risk.view(-1, 1), templates["anti1"].view(1, -1), weights)
    return weights


def router_weights(probs: torch.Tensor, router: str) -> torch.Tensor:
    if router == "dynamic":
        return compute_dynamic_weights(probs)
    if router == "rule":
        return compute_rule_weights(probs)
    if router == "average":
        return torch.full((probs.shape[0], len(EXPERTS)), 1.0 / len(EXPERTS), dtype=probs.dtype, device=probs.device)
    static = torch.tensor([expert["static_weight"] for expert in EXPERTS], dtype=probs.dtype, device=probs.device)
    static = static / static.sum()
    return static.view(1, -1).repeat(probs.shape[0], 1)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    idx_path = Path(args.testdata) / "test_B_images.idx3-ubyte"
    if not idx_path.exists():
        print(f"错误：找不到测试图像文件 {idx_path}", file=sys.stderr)
        sys.exit(1)

    images = load_idx_images(idx_path)
    root = Path(__file__).resolve().parent
    n = images.shape[0]
    print(f"router={args.router} device={device} samples={n} tta={args.tta_n}", flush=True)
    print("experts=" + ", ".join(expert["name"] for expert in EXPERTS), flush=True)

    expert_probs = []
    for expert in EXPERTS:
        print(f"[expert] {expert['name']}", flush=True)
        expert_probs.append(predict_expert_probabilities(root, expert, images, device, args.batch_size, args.tta_n).unsqueeze(1))
    probs = torch.cat(expert_probs, dim=1)
    weights = router_weights(probs, args.router)
    final_probs = (weights.unsqueeze(-1) * probs).sum(dim=1)
    preds = final_probs.argmax(dim=1).tolist()
    confs = final_probs.max(dim=1).values.tolist()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pred in preds:
            f.write(f"{pred}\n")

    print(f"mean_weights={[round(v, 4) for v in weights.mean(dim=0).tolist()]}", flush=True)
    print(f"weight_std={[round(v, 4) for v in weights.std(dim=0).tolist()]}", flush=True)
    print(f"mean_confidence={sum(confs) / max(1, len(confs)):.4f}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
