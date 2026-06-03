"""Exact routing strategy evaluation on new TestA — uses original code logic."""

import struct, sys
from pathlib import Path
import torch, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from predict import MediumCNN, PreActResNetTiny, build_model_from_checkpoint, load_idx_images, predict_with_tta

EXPERTS = [
    {"name": "wide_resnet_tiny_raw_seed42", "model_name": "wide_resnet_tiny", "weight": 0.7},
    {"name": "medium_anti1_seed2026",       "model_name": "medium_cnn",        "weight": 0.2},
    {"name": "medium_raw_seed3407",         "model_name": "medium_cnn",        "weight": 0.1},
]
TEMPLATES_4 = torch.tensor([[0.70,0.20,0.10],[0.35,0.35,0.30],[0.50,0.30,0.20],[0.45,0.45,0.10]], dtype=torch.float32)
BASE_W = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float32)


def compute_expert_probs(device, batch_size=256, tta_n=8):
    data = PROJECT_ROOT / "build/testA_eval"
    images = load_idx_images(data / "test_B_images.idx3-ubyte")
    n = images.shape[0]
    print(f"计算专家概率: {n} 样本, device={device}")

    models_dir = PROJECT_ROOT / "build/submission/models"
    all_probs = []
    for expert in EXPERTS:
        fold_sum = None
        for fold in range(5):
            ckpt = models_dir / expert["name"] / f"fold_{fold}" / "testa_specialist_best.pt"
            model = build_model_from_checkpoint(ckpt, device)
            batches = [predict_with_tta(model, images[s:s+batch_size], device, tta_n)
                       for s in range(0, n, batch_size)]
            fold_res = torch.cat(batches, dim=0)
            del model
            if device == "cuda": torch.cuda.empty_cache()
            fold_sum = fold_res if fold_sum is None else fold_sum + fold_res
        all_probs.append((fold_sum / 5.0).unsqueeze(1))
    return torch.cat(all_probs, dim=1)  # (N, 3, 10)


def load_labels():
    with open(PROJECT_ROOT / "build/testA_eval/test_A_labels.idx1-ubyte", "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()


# ─── Exact Dynamic MoE Router logic (from search_dynamic_moe_router_oof.py) ───

def dynamic_moe_predict(probs, params_dict):
    device = probs.device
    base_w = torch.tensor([0.7, 0.2, 0.1], device=device)
    params = torch.tensor([[params_dict["confidence"], params_dict["margin"],
                            params_dict["disagreement"], params_dict["anti1_boost"]]],
                          device=device, dtype=torch.float32)

    # Features (exact copy of compute_features)
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[..., 0]                                    # (N, 3)
    margin = top2[..., 0] - top2[..., 1]                         # (N, 3)
    mean_dist = (probs * base_w.view(1, -1, 1)).sum(dim=1, keepdim=True).clamp_min(1e-9)
    disagreement = (probs.clamp_min(1e-9) * (probs.clamp_min(1e-9) / mean_dist).log()).sum(dim=-1)  # (N, 3)
    # Center
    confidence = confidence - confidence.mean(dim=1, keepdim=True)
    margin = margin - margin.mean(dim=1, keepdim=True)
    disagreement = disagreement - disagreement.mean(dim=1, keepdim=True)
    # anti1 signal
    fixed_probs = (probs * base_w.view(1, -1, 1)).sum(dim=1)
    fixed_pred = fixed_probs.argmax(dim=1)
    anti1_signal = (fixed_pred == 1).float().view(-1, 1) * torch.tensor([[0.,1.,0.]], device=device)

    # Score and weight (exact copy of predict_for_params)
    log_base = base_w.log().view(1, 1, -1)
    score = (log_base
             + params[0, 0] * confidence.unsqueeze(0)
             + params[0, 1] * margin.unsqueeze(0)
             - params[0, 2] * disagreement.unsqueeze(0)
             + params[0, 3] * anti1_signal.unsqueeze(0))
    weights = torch.softmax(score, dim=-1)  # (1, N, 3)
    return (weights.squeeze(0).unsqueeze(-1) * probs).sum(dim=1)


# ─── Exact Domain-Aware Rule Router logic (from search_domain_aware_rule_router.py) ───

def domain_rule_predict(probs, params_row, templates):
    """Exact per-sample rule routing."""
    device = probs.device
    n = probs.shape[0]
    templates = templates.to(device)
    testa_w = templates[0]  # [0.7, 0.2, 0.1]
    fixed_probs = (probs * testa_w.view(1, -1, 1)).sum(dim=1)
    fixed_pred = fixed_probs.argmax(dim=1)

    max_probs, expert_preds = probs.max(dim=-1)  # (N, 3)
    wide_conf = max_probs[:, 0]
    medium_conf = (max_probs[:, 1] + max_probs[:, 2]) * 0.5
    medium_agree = (expert_preds[:, 1] == expert_preds[:, 2])
    wide_agrees_medium = (expert_preds[:, 0] == expert_preds[:, 1]) | (expert_preds[:, 0] == expert_preds[:, 2])
    anti1_pred = expert_preds[:, 1]
    anti1_delta = fixed_probs[:, 1] - probs[:, 1, 1]

    # Rule masks
    anti1_mask = (fixed_pred == 1) & (anti1_pred != 1) & (anti1_delta >= params_row["anti1_delta"])
    mnist_mask = medium_agree & (medium_conf >= params_row["medium_conf"]) & (wide_conf <= medium_conf + params_row["wide_slack"])
    testa_mask = (wide_conf >= params_row["wide_conf"]) & wide_agrees_medium

    # Template selection (balanced=2 is default)
    template_idx = torch.full((n,), 2, dtype=torch.long, device=device)
    template_idx = torch.where(testa_mask, torch.tensor(0, device=device), template_idx)
    template_idx = torch.where(mnist_mask, torch.tensor(1, device=device), template_idx)
    template_idx = torch.where(anti1_mask, torch.tensor(3, device=device), template_idx)

    weights = templates[template_idx].unsqueeze(-1)
    return (weights * probs).sum(dim=1)


def acc(probs, labels):
    return (probs.argmax(dim=1) == labels).float().mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    print(f"设备: {device}")

    cache = PROJECT_ROOT / "build/testA_eval/expert_probs.pt"
    if cache.exists():
        print("加载缓存概率...")
        probs = torch.load(cache, map_location=device, weights_only=False)
        labels = load_labels()
    else:
        probs = compute_expert_probs(device)
        labels = load_labels()
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(probs.cpu(), cache)
        print(f"概率缓存已保存: {cache}")

    results = {}

    # 1. Fixed
    a = acc((probs * torch.tensor([0.7,0.2,0.1]).view(1,3,1)).sum(dim=1), labels)
    results["固定权重 0.7/0.2/0.1"] = a
    print(f"\n=== 结果 ===\n  固定权重: {a:.4%}")

    # 2. Dynamic MoE — best params
    best = {"confidence": 0.75, "margin": 0.0, "disagreement": -1.0, "anti1_boost": 0.0}
    a = acc(dynamic_moe_predict(probs, best), labels)
    results["Dynamic MoE (conf=0.75)"] = a
    print(f"  Dynamic MoE: {a:.4%}")

    # Also try other top params
    for rank, params in enumerate([
        {"confidence": 5.0, "margin": 1.5, "disagreement": 0.0, "anti1_boost": 0.0},
        {"confidence": -1.0, "margin": 0.75, "disagreement": -1.0, "anti1_boost": 0.0},
        {"confidence": 2.5, "margin": -1.0, "disagreement": -1.0, "anti1_boost": 0.0},
    ], start=2):
        a = acc(dynamic_moe_predict(probs, params), labels)
        results[f"Dynamic MoE rank{rank}"] = a
        print(f"  Dynamic MoE rank{rank}: {a:.4%}")

    # 3. Domain-Aware Rule Router — best params
    rule_params = {"medium_conf": 0.7, "wide_conf": 0.7, "anti1_delta": 0.15, "wide_slack": 0.1}
    a = acc(domain_rule_predict(probs, rule_params, TEMPLATES_4), labels)
    results["Rule Router (best)"] = a
    print(f"  Rule Router: {a:.4%}")

    # 4. Static templates
    for name, w in [("testa [.7,.2,.1]", [0.7,0.2,0.1]), ("mnist [.35,.35,.30]", [0.35,0.35,0.30]),
                     ("balanced [.5,.3,.2]", [0.5,0.3,0.2]), ("anti1 [.45,.45,.1]", [0.45,0.45,0.1])]:
        a = acc((probs * torch.tensor(w).view(1,3,1)).sum(dim=1), labels)
        results[name] = a
        print(f"  {name}: {a:.4%}")

    print("\n=== 排序 ===")
    for name, a in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(a * 50)
        print(f"  {a:.4%} {bar} {name}")


if __name__ == "__main__":
    main()
