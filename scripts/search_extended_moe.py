"""Extended Dynamic MoE: N experts, 4-parameter linear gate, multi-domain evaluation.

Differs from the 3-expert version:
- Supports arbitrary number of experts
- anti1_signal is configurable per-expert (not hardcoded [0,1,0])
- Evaluates on multi-domain data, not just old TestA OOF
"""

from __future__ import annotations

import json, sys
from itertools import product
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Expert registry ──
# Each expert: label, base_weight (from OOF acc), anti1 flag, OOF path
EXPERTS = [
    {
        "label": "wide_resnet_tiny",
        "base_weight": 0.7797,  # OOF accuracy normalized
        "anti1": False,
        "oof": PROJECT_ROOT / "outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_raw_seed777",
        "base_weight": 0.7449,
        "anti1": False,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_raw_seed777_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_raw_seed3407",
        "base_weight": 0.7471,
        "anti1": False,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_raw_seed3407_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_anti1_seed2026",
        "base_weight": 0.7451,
        "anti1": True,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "partial_init_mixup",
        "base_weight": 0.7420,
        "anti1": False,
        "oof": PROJECT_ROOT / "outputs_runs/testa_partial_init_lr1e4_mixup01_erasing005_e40/oof/oof_probabilities.pt",
    },
]

# Also load the 3-expert domain cache for multi-domain evaluation
DOMAIN_CACHE = PROJECT_ROOT / "outputs_runs/domain_aware_rule_router/cache"
DOMAIN_FILES = {
    "TestA_old": None,  # from OOF
    "MNIST-family": DOMAIN_CACHE / "mnist_family.pt",
    "MNIST-C": DOMAIN_CACHE / "mnist_c.pt",
    "local_external": DOMAIN_CACHE / "external_digits.pt",
}

PARAM_GRID = {
    "confidence": [-2.0, -1.0, 0.0, 0.75, 1.5, 2.5, 3.5, 5.0],
    "margin": [-1.0, 0.0, 0.75, 1.5, 2.5, 3.5],
    "disagreement": [-1.0, 0.0, 0.5, 1.0, 1.75, 2.5, 3.5],
    "anti1_boost": [0.0, 0.25, 0.5, 1.0, 1.5],
}


def load_oof_probs(device):
    """Load old TestA OOF probs for all experts: (N, E, 10)"""
    probs_list, labels = [], None
    for exp in EXPERTS:
        data = torch.load(exp["oof"], map_location="cpu", weights_only=False)
        if labels is None:
            labels = data["labels"].long()
        probs_list.append(data["probabilities"].float())
    E = len(EXPERTS)
    N = labels.shape[0]
    return torch.stack(probs_list, dim=1).to(device), labels.to(device)  # (N, E, 10)


def load_multi_domain(device):
    """Load multi-domain data. Returns list of (name, probs, labels).

    Note: domain cache only has 3 experts. For now, use domain data only
    for the 3-expert subset evaluation. The full N-expert evaluation
    happens on OOF + new TestA.
    """
    domains = []
    for name, path in DOMAIN_FILES.items():
        if path is None:  # TestA_old via OOF
            continue
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        probs = data["probabilities"]  # (n, 3, 10)
        labels = data["labels"].long()
        domains.append((name, probs, labels))
    return domains


def compute_features(probs, base_w, anti1_vector):
    """probs: (N, E, 10), base_w: (E,), anti1_vector: (E,) bool"""
    device = probs.device
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[..., 0]                                    # (N, E)
    margin = top2[..., 0] - top2[..., 1]                         # (N, E)
    # Weighted mean distribution
    mean_dist = (probs * base_w.view(1, -1, 1)).sum(dim=1, keepdim=True).clamp_min(1e-9)
    disagreement = (probs.clamp_min(1e-9) * (probs.clamp_min(1e-9) / mean_dist).log()).sum(dim=-1)  # (N, E)
    # Center
    confidence = confidence - confidence.mean(dim=1, keepdim=True)
    margin = margin - margin.mean(dim=1, keepdim=True)
    disagreement = disagreement - disagreement.mean(dim=1, keepdim=True)
    # anti1: per-expert flag
    fixed_probs = (probs * base_w.view(1, -1, 1)).sum(dim=1)
    fixed_pred = fixed_probs.argmax(dim=1)
    anti1_signal = (fixed_pred == 1).float().view(-1, 1) * anti1_vector.float().view(1, -1)
    return confidence, margin, disagreement, anti1_signal


def predict(probs, base_w, features, params):
    """probs: (N, E, 10), params: (G, 4), returns predictions (G, N)"""
    E = probs.shape[1]
    confidence, margin, disagreement, anti1_signal = features
    log_base = base_w.log().view(1, 1, -1)
    G = params.shape[0]

    all_preds = []
    chunk_size = 64
    for start in range(0, G, chunk_size):
        chunk = params[start : start + chunk_size]  # (c, 4)
        score = (log_base
                 + chunk[:, 0].view(-1, 1, 1) * confidence.unsqueeze(0)
                 + chunk[:, 1].view(-1, 1, 1) * margin.unsqueeze(0)
                 - chunk[:, 2].view(-1, 1, 1) * disagreement.unsqueeze(0)
                 + chunk[:, 3].view(-1, 1, 1) * anti1_signal.unsqueeze(0))
        weights = torch.softmax(score, dim=-1)  # (c, N, E)
        ensemble_probs = torch.einsum("gne,nec->gnc", weights, probs)
        all_preds.append(ensemble_probs.argmax(dim=-1))
    return torch.cat(all_preds, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}, 专家数: {len(EXPERTS)}")

    # Load OOF data
    oof_probs, oof_labels = load_oof_probs(device)
    print(f"OOF: {oof_probs.shape}")

    # Also load new TestA probs if cached
    new_testa_cache = PROJECT_ROOT / "build/testA_eval/expert_probs.pt"
    has_new_testa = new_testa_cache.exists()

    # Build base weights and anti1 vector
    raw_weights = torch.tensor([e["base_weight"] for e in EXPERTS], device=device)
    base_w = raw_weights / raw_weights.sum()
    anti1_vec = torch.tensor([e["anti1"] for e in EXPERTS], device=device)
    print(f"base_weights: {[f'{w:.4f}' for w in base_w.tolist()]}")
    print(f"anti1_vec: {anti1_vec.tolist()}")

    # Features
    features = compute_features(oof_probs, base_w, anti1_vec)

    # Grid search params
    param_rows = list(product(PARAM_GRID["confidence"], PARAM_GRID["margin"],
                              PARAM_GRID["disagreement"], PARAM_GRID["anti1_boost"]))
    params = torch.tensor(param_rows, device=device, dtype=torch.float32)
    print(f"参数组合: {len(param_rows)}")

    # Evaluate on OOF
    preds = predict(oof_probs, base_w, features, params)  # (G, N)
    oof_acc = (preds == oof_labels.view(1, -1)).float().mean(dim=1)

    # Fixed baseline
    fixed_pred = (oof_probs * base_w.view(1, -1, 1)).sum(dim=1).argmax(dim=1)
    fixed_acc = (fixed_pred == oof_labels).float().mean().item()
    print(f"\n固定权重 (OOF归一化) 旧TestA: {fixed_acc:.4%}")

    # Current 3-expert fixed [0.7, 0.2, 0.1]
    w3 = torch.tensor([0.7, 0.0, 0.2, 0.1, 0.0], device=device)  # 5 experts, only 3 active
    w3 = w3 / w3.sum()
    fixed3_pred = (oof_probs * w3.view(1, -1, 1)).sum(dim=1).argmax(dim=1)
    fixed3_acc = (fixed3_pred == oof_labels).float().mean().item()
    print(f"固定权重 (0.7/0/0.2/0.1/0) 旧TestA: {fixed3_acc:.4%}")

    # Top Dynamic MoE
    topk = min(20, len(param_rows))
    top_vals, top_idx = torch.topk(oof_acc, k=topk)
    print(f"\n=== Top-{topk} Dynamic MoE (旧TestA OOF) ===")
    for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), 1):
        row = param_rows[idx]
        print(f"  #{rank}: acc={val:.4%} conf={row[0]:.2f} margin={row[1]:.2f} dis={row[2]:.2f} anti1={row[3]:.2f}")

    # Evaluate on multi-domain (3-expert subset only — domain cache limitation)
    multi_domains = load_multi_domain(device)
    if multi_domains:
        print(f"\n=== 多域评估 (仅3专家子集) ===")
        # Build 3-expert version without full re-computation
        # Use the already-cached 3-expert probs from domain cache
        for name, probs, labels in multi_domains:
            # Fixed
            w3_dom = torch.tensor([0.7, 0.2, 0.1], device=device)
            fp = (probs.to(device) * w3_dom.view(1, -1, 1)).sum(dim=1).argmax(dim=1)
            fa = (fp == labels.to(device)).float().mean().item()
            # Best template
            w_mnist = torch.tensor([0.35, 0.35, 0.30], device=device)
            mp = (probs.to(device) * w_mnist.view(1, -1, 1)).sum(dim=1).argmax(dim=1)
            ma = (mp == labels.to(device)).float().mean().item()
            print(f"  {name}: fixed={fa:.4%} mnist={ma:.4%}")

    # New TestA evaluation
    if has_new_testa:
        import struct
        new_probs_all = torch.load(new_testa_cache, map_location=device, weights_only=False)  # (3500, 3, 10)
        with open(PROJECT_ROOT / "build/testA_eval/test_A_labels.idx1-ubyte", "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            new_labels = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long().to(device)

        print(f"\n=== 新 TestA 评估 ===")
        # Fixed 3-expert
        fp3 = (new_probs_all * w3_dom.view(1, -1, 1)).sum(dim=1).argmax(dim=1)
        fa3 = (fp3 == new_labels).float().mean().item()
        print(f"  固定 0.7/0.2/0.1: {fa3:.4%}")

        # mnist template
        mp3 = (new_probs_all * w_mnist.view(1, -1, 1)).sum(dim=1).argmax(dim=1)
        ma3 = (mp3 == new_labels).float().mean().item()
        print(f"  mnist 模板: {ma3:.4%}")

    # Summary
    print(f"\n=== 结论 ===")
    print(f"  5专家 OOF归一化固定权重: {fixed_acc:.4%}")
    print(f"  3专家 0.7/0.2/0.1 固定权重: {fixed3_acc:.4%}")
    best_oof = top_vals[0].item()
    print(f"  5专家 Dynamic MoE best: {best_oof:.4%}")
    print(f"  增益: {best_oof - fixed_acc:+.4%}")


if __name__ == "__main__":
    main()
