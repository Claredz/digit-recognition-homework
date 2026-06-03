"""Compare all routing strategies on new TestA data with labels.

Uses the same 15 checkpoints as the submission package, computes per-expert
probabilities once, then applies each strategy and reports accuracy.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Re-use model classes from predict.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from predict import (
    MediumCNN, PreActResNetTiny, build_model_from_checkpoint,
    load_idx_images, predict_with_tta,
)
EXPERTS = [
    {"name": "wide_resnet_tiny_raw_seed42", "model_name": "wide_resnet_tiny", "weight": 0.7},
    {"name": "medium_anti1_seed2026",       "model_name": "medium_cnn",        "weight": 0.2},
    {"name": "medium_raw_seed3407",         "model_name": "medium_cnn",        "weight": 0.1},
]

# Templates for domain-aware rule router
TEMPLATES = {
    "testa":    [0.70, 0.20, 0.10],
    "mnist":    [0.35, 0.35, 0.30],
    "balanced": [0.50, 0.30, 0.20],
    "anti1":    [0.45, 0.45, 0.10],
}

# Best dynamic MoE router params
DYNAMIC_MOE_BEST = {"confidence": 0.75, "margin": 0.0, "disagreement": -1.0, "anti1_boost": 0.0}

# Learned router mean TestA weights (from best_by_objective.hidden_b_balanced)
LEARNED_WEIGHTS = {
    "hidden_b_balanced": [0.4604, 0.2954, 0.2442],
    "hidden_b_easy":     [0.4557, 0.2987, 0.2457],
    "hidden_b_hard":     [0.4541, 0.2968, 0.2491],
}


def compute_expert_probs(data_dir: str, device: str, batch_size: int, tta_n: int):
    """Compute (N, 3, 10) tensor: per-expert 5-fold-averaged probabilities."""
    testdata = Path(data_dir)
    idx_path = testdata / "test_B_images.idx3-ubyte"
    images = load_idx_images(idx_path)
    n = images.shape[0]
    print(f"计算专家概率: {n} 样本, device={device}, TTA={tta_n}")

    project_root = Path(__file__).resolve().parents[1]
    models_root = project_root / "build" / "submission" / "models"
    expert_probs_list = []

    for ei, expert in enumerate(EXPERTS):
        fold_probs = None
        for fold in range(5):
            ckpt = models_root / expert["name"] / f"fold_{fold}" / "testa_specialist_best.pt"
            model = build_model_from_checkpoint(ckpt, device)
            batch_list = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                probs = predict_with_tta(model, images[start:end], device, tta_n)
                batch_list.append(probs)
            fold_result = torch.cat(batch_list, dim=0)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            fold_probs = fold_result if fold_probs is None else fold_probs + fold_result
        expert_probs_list.append((fold_probs / 5.0).unsqueeze(1))
        print(f"  [{ei+1}/3] {expert['name']} 完成")

    all_probs = torch.cat(expert_probs_list, dim=1)  # (N, 3, 10)
    return all_probs


def load_labels(data_dir: str):
    lbl_path = Path(data_dir) / "test_A_labels.idx1-ubyte"
    with open(lbl_path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8).copy()
    return torch.from_numpy(labels).long()


def accuracy(probs, labels):
    return (probs.argmax(dim=1) == labels).float().mean().item()


# --- Routing strategies ---

def strategy_fixed(probs):
    """0.7/0.2/0.1 fixed weights."""
    w = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32).view(1, 3, 1)
    return (probs * w).sum(dim=1)


def strategy_template(probs, weights):
    """Apply a static weight template."""
    w = torch.tensor(weights, dtype=torch.float32).view(1, 3, 1)
    return (probs * w).sum(dim=1)


def strategy_dynamic_moe(probs, params):
    """Per-sample dynamic weight adjustment based on confidence/margin/disagreement."""
    # Base weights
    base = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32)
    n = probs.shape[0]

    # Features per expert
    max_prob = probs.max(dim=2).values  # (N, 3)
    top2 = probs.topk(2, dim=2).values
    margin = top2[:, :, 0] - top2[:, :, 1]  # (N, 3)

    # Disagreement: mean JS divergence between expert pairs
    eps = 1e-8
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    m01 = 0.5 * (p0 + p1)
    js01 = 0.5 * ((p0 * (p0 / (m01 + eps)).log()).sum(1) + (p1 * (p1 / (m01 + eps)).log()).sum(1))
    m02 = 0.5 * (p0 + p2)
    js02 = 0.5 * ((p0 * (p0 / (m02 + eps)).log()).sum(1) + (p2 * (p2 / (m02 + eps)).log()).sum(1))
    m12 = 0.5 * (p1 + p2)
    js12 = 0.5 * ((p1 * (p1 / (m12 + eps)).log()).sum(1) + (p2 * (p2 / (m12 + eps)).log()).sum(1))
    disagreement = (js01 + js02 + js12) / 3.0  # (N,)

    # Apply routing adjustments (from search logic: adjust base weights per sample)
    conf = params["confidence"]
    margin_p = params["margin"]
    disagree_p = params["disagreement"]
    anti1 = params["anti1_boost"]

    # Per-sample weight deltas
    deltas = torch.zeros(n, 3)

    # Confidence: boost experts with high confidence
    for e in range(3):
        deltas[:, e] += conf * (max_prob[:, e] - max_prob.mean(dim=1))

    # Margin: boost experts with large margin
    for e in range(3):
        deltas[:, e] += margin_p * (margin[:, e] - margin.mean(dim=1))

    # Disagreement: when experts disagree, shift toward wide_resnet (expert 0)
    disagree_score = disagreement - disagreement.mean()
    deltas[:, 0] += disagree_p * disagree_score
    deltas[:, 1] -= 0.5 * disagree_p * disagree_score
    deltas[:, 2] -= 0.5 * disagree_p * disagree_score

    # Anti1 boost: boost anti1 expert (expert 1) when class-1 probability is high
    p_class1 = probs[:, :, 1]  # (N, 3) probability of class 1 per expert
    anti1_signal = p_class1[:, 1] - p_class1.mean(dim=1)  # expert 1's class1 prob vs mean
    deltas[:, 1] += anti1 * anti1_signal

    # Apply softmax to get final weights
    logits = base.view(1, 3).log() + deltas
    weights = torch.softmax(logits, dim=1).unsqueeze(2)

    return (probs * weights).sum(dim=1)


def main():
    data_dir = str(PROJECT_ROOT / "build" / "testA_eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    print(f"设备: {device}")
    probs = compute_expert_probs(data_dir, device, batch_size=256, tta_n=8)
    labels = load_labels(data_dir)

    results = {}

    # 1. Fixed weight (baseline)
    print("\n=== 策略对比 ===")
    acc = accuracy(strategy_fixed(probs), labels)
    results["固定权重 0.7/0.2/0.1"] = acc
    print(f"  固定权重 0.7/0.2/0.1: {acc:.4%}")

    # 2. Static templates (domain-aware)
    for name, w in TEMPLATES.items():
        acc = accuracy(strategy_template(probs, w), labels)
        label = f"静态模板 {name} ({w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f})"
        results[label] = acc
        print(f"  {label}: {acc:.4%}")

    # 3. Dynamic MoE Router
    acc = accuracy(strategy_dynamic_moe(probs, DYNAMIC_MOE_BEST), labels)
    results["动态MoE (conf=0.75,margin=0,dis=-1)"] = acc
    print(f"  动态MoE (conf=0.75,margin=0,dis=-1): {acc:.4%}")

    # 4. Learned router mean weights
    for scenario, w in LEARNED_WEIGHTS.items():
        acc = accuracy(strategy_template(probs, w), labels)
        label = f"学习路由器 {scenario} ({w[0]:.4f}/{w[1]:.4f}/{w[2]:.4f})"
        results[label] = acc
        print(f"  {label}: {acc:.4%}")

    # Summary
    print("\n=== 排序 ===")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(acc * 50)
        print(f"  {acc:.4%}  {bar}  {name}")


if __name__ == "__main__":
    main()
