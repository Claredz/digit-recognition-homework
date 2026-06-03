"""Dynamic MoE with 5 diverse experts on new TestA.

Experts: wide_resnet, medium_anti1, medium_raw, robust_v1, MNIST_clean
"""

import struct, sys, json
from itertools import product
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from predict import MediumCNN, PreActResNetTiny, PreActBlock, LargeCNN, load_idx_images, predict_with_tta, build_model_from_checkpoint

EXPERTS_5 = [
    {"label": "wide_resnet",     "base_w": 0.7797, "anti1": False, "type": "specialist",
     "pattern": "outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/seed_42/fold_{fold}/checkpoints/testa_specialist_best.pt", "folds": 5},
    {"label": "medium_anti1",    "base_w": 0.7451, "anti1": True,  "type": "specialist",
     "pattern": "outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/seed_2026/fold_{fold}/checkpoints/testa_specialist_best.pt", "folds": 5},
    {"label": "medium_raw",      "base_w": 0.7471, "anti1": False, "type": "specialist",
     "pattern": "outputs_runs/testa_medium_v2_raw_seed3407_e60/seed_3407/fold_{fold}/checkpoints/testa_specialist_best.pt", "folds": 5},
    {"label": "robust_v1",       "base_w": 0.9280, "anti1": False, "type": "single",
     "ckpt": "outputs_submission/checkpoints/robust_expert_best.pt"},
    {"label": "MNIST_clean",     "base_w": 0.7657, "anti1": False, "type": "single",
     "ckpt": "outputs_submission/checkpoints/best_model_state.pt"},
]

PARAM_GRID = {
    "confidence":    [-2.0, -1.0, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0],
    "margin":        [-1.0, 0.0, 0.5, 0.75, 1.0, 1.5, 2.5, 3.5],
    "disagreement":  [-1.0, 0.0, 0.5, 1.0, 1.75, 2.5, 3.5],
    "anti1_boost":   [0.0, 0.1, 0.25, 0.5, 1.0, 1.5],
}


def load_single_model(ckpt, device):
    cp = torch.load(ckpt, map_location=device, weights_only=False)
    mn = cp.get("model_name", "medium_cnn")
    dropout = cp.get("config", {}).get("dropout", 0.3)
    if mn == "medium_cnn":
        model = MediumCNN(dropout=dropout)
    elif mn == "wide_resnet_tiny":
        model = PreActResNetTiny(dropout=dropout, widths=(48, 96, 192))
    else:
        raise ValueError(f"Unknown: {mn}")
    model.load_state_dict(cp["model_state_dict"])
    model.to(device).eval()
    return model


def compute_all_probs(device):
    """Compute (N, 5, 10) probs for all experts on new TestA."""
    images = load_idx_images(PROJECT_ROOT / "build/testA_eval/test_B_images.idx3-ubyte")
    n = images.shape[0]
    all_probs = []

    for exp in EXPERTS_5:
        print(f"  [{exp['label']}] ", end="", flush=True)
        if exp["type"] == "specialist":
            fold_sum = None
            for fold in range(exp["folds"]):
                ckpt = PROJECT_ROOT / exp["pattern"].format(fold=fold)
                model = build_model_from_checkpoint(ckpt, device)
                batches = [predict_with_tta(model, images[s:s+256], device, 8)
                           for s in range(0, n, 256)]
                fp = torch.cat(batches, dim=0)
                del model
                if device == "cuda": torch.cuda.empty_cache()
                fold_sum = fp if fold_sum is None else fold_sum + fp
            probs = fold_sum / exp["folds"]
        else:
            model = load_single_model(PROJECT_ROOT / exp["ckpt"], device)
            batches = [predict_with_tta(model, images[s:s+256], device, 8)
                       for s in range(0, n, 256)]
            probs = torch.cat(batches, dim=0)
            del model
            if device == "cuda": torch.cuda.empty_cache()
        all_probs.append(probs.cpu())
        print(f"done", flush=True)

    return torch.stack(all_probs, dim=1)  # (N, 5, 10)


def load_labels():
    with open(PROJECT_ROOT / "build/testA_eval/test_A_labels.idx1-ubyte", "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()


def compute_features(probs, base_w, anti1_vec):
    """probs: (N, E, 10), base_w: (E,), anti1_vec: (E,) bool"""
    E = probs.shape[1]
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]
    mean_dist = (probs * base_w.view(1, -1, 1)).sum(dim=1, keepdim=True).clamp_min(1e-9)
    disagreement = (probs.clamp_min(1e-9) * (probs.clamp_min(1e-9) / mean_dist).log()).sum(dim=-1)

    confidence = confidence - confidence.mean(dim=1, keepdim=True)
    margin = margin - margin.mean(dim=1, keepdim=True)
    disagreement = disagreement - disagreement.mean(dim=1, keepdim=True)

    fixed_probs = (probs * base_w.view(1, -1, 1)).sum(dim=1)
    fixed_pred = fixed_probs.argmax(dim=1)
    anti1_signal = (fixed_pred == 1).float().view(-1, 1) * anti1_vec.float().view(1, -1)
    return confidence, margin, disagreement, anti1_signal


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    print(f"设备: {device}")

    # Check cache
    cache_path = PROJECT_ROOT / "build/testA_eval/hybrid_5_probs.pt"
    if cache_path.exists():
        print("加载缓存...")
        probs = torch.load(cache_path, map_location='cpu', weights_only=False)
        labels = load_labels()
    else:
        print("计算 5 专家概率...")
        probs = compute_all_probs(device)
        labels = load_labels()
        torch.save(probs, cache_path)

    probs_dev = probs.to(device)
    labels_dev = labels.to(device)
    N, E, C = probs.shape
    print(f"数据: {N} 样本, {E} 专家, {C} 类")

    # Base weights from per-expert accuracies
    base_w_raw = torch.tensor([e["base_w"] for e in EXPERTS_5], device=device)
    base_w = base_w_raw / base_w_raw.sum()
    anti1_vec = torch.tensor([e["anti1"] for e in EXPERTS_5], device=device)

    # Features
    features = compute_features(probs_dev, base_w, anti1_vec)

    # Fixed baseline
    fixed_p = (probs_dev * base_w.view(1, -1, 1)).sum(dim=1).argmax(dim=1)
    fixed_acc = (fixed_p == labels_dev).float().mean().item()
    print(f"\n基线 (OOF归一化固定权重): {fixed_acc:.4%}")

    # Grid search
    param_rows = list(product(PARAM_GRID["confidence"], PARAM_GRID["margin"],
                              PARAM_GRID["disagreement"], PARAM_GRID["anti1_boost"]))
    params = torch.tensor(param_rows, device=device, dtype=torch.float32)
    G = len(param_rows)
    print(f"搜索 {G} 种参数组合...")

    # Evaluate in chunks
    all_accs = []
    chunk = 128
    for start in range(0, G, chunk):
        end = min(start + chunk, G)
        p_chunk = params[start:end]
        score = (base_w.log().view(1, 1, -1)
                 + p_chunk[:, 0].view(-1, 1, 1) * features[0].unsqueeze(0)
                 + p_chunk[:, 1].view(-1, 1, 1) * features[1].unsqueeze(0)
                 - p_chunk[:, 2].view(-1, 1, 1) * features[2].unsqueeze(0)
                 + p_chunk[:, 3].view(-1, 1, 1) * features[3].unsqueeze(0))
        weights = torch.softmax(score, dim=-1)
        ens = torch.einsum("gne,nec->gnc", weights, probs_dev)
        preds = ens.argmax(dim=-1)
        accs = (preds == labels_dev.view(1, -1)).float().mean(dim=1)
        all_accs.append(accs)

    all_accs = torch.cat(all_accs, dim=0)

    # Top results
    topk = min(30, G)
    top_vals, top_idx = torch.topk(all_accs, k=topk)

    print(f"\n=== Dynamic MoE Top-{topk} ===")
    for rank in range(topk):
        idx = top_idx[rank].item()
        val = top_vals[rank].item()
        row = param_rows[idx]
        # Get mean weights
        score = (base_w.log().view(1, 1, -1)
                 + params[idx, 0] * features[0].unsqueeze(0)
                 + params[idx, 1] * features[1].unsqueeze(0)
                 - params[idx, 2] * features[2].unsqueeze(0)
                 + params[idx, 3] * features[3].unsqueeze(0))
        w = torch.softmax(score, dim=-1).squeeze(0).mean(dim=0)
        w_str = "/".join(f"{x:.3f}" for x in w.cpu().tolist())
        if rank < 10:
            print(f"  #{rank+1}: {val:.4%} conf={row[0]:.2f} margin={row[1]:.2f} "
                  f"dis={row[2]:.2f} anti1={row[3]:.2f}  mean_w=[{w_str}]")

    # Compare with previous bests
    print(f"\n=== 对比汇总 ===")
    print(f"  3专家固定 0.70/0.20/0.10:          93.40%")
    print(f"  3专家最佳静态 0.40/0.10/0.50:      94.14%")
    print(f"  5专家固定(OOF归一化):              {fixed_acc:.4%}")
    print(f"  5专家 Dynamic MoE best:            {top_vals[0].item():.4%}")
    print(f"  Oracle (5专家上限):                 97.34%")

    best_gain = top_vals[0].item() - 0.9414
    print(f"\n  vs 3专家最佳静态: {best_gain:+.4%}")

    # Save best params
    best_idx = top_idx[0].item()
    best_row = param_rows[best_idx]
    print(f"\n最佳参数: conf={best_row[0]:.3f}, margin={best_row[1]:.3f}, "
          f"dis={best_row[2]:.3f}, anti1={best_row[3]:.3f}")


if __name__ == "__main__":
    main()
