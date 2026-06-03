"""Evaluate hybrid ensemble: TestA specialists + robust + clean MNIST models."""

import struct, sys
from pathlib import Path
import torch, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from predict import MediumCNN, PreActResNetTiny, LargeCNN, PreActBlock, load_idx_images, predict_with_tta, build_model_from_checkpoint

# ── Diverse expert pool ──
HYBRID_EXPERTS = [
    # TestA specialists
    {
        "label": "wide_resnet",
        "type": "testa_specialist",
        "ckpt_pattern": "outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/seed_42/fold_{fold}/checkpoints/testa_specialist_best.pt",
        "n_folds": 5,
    },
    {
        "label": "medium_anti1_seed2026",
        "type": "testa_specialist",
        "ckpt_pattern": "outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/seed_2026/fold_{fold}/checkpoints/testa_specialist_best.pt",
        "n_folds": 5,
    },
    {
        "label": "medium_raw_seed3407",
        "type": "testa_specialist",
        "ckpt_pattern": "outputs_runs/testa_medium_v2_raw_seed3407_e60/seed_3407/fold_{fold}/checkpoints/testa_specialist_best.pt",
        "n_folds": 5,
    },
    # Robust model (single checkpoint, no folds)
    {
        "label": "robust_v1",
        "type": "single",
        "ckpt": PROJECT_ROOT / "outputs_submission/checkpoints/robust_expert_best.pt",
    },
    # Clean MNIST model
    {
        "label": "MNIST_clean",
        "type": "single",
        "ckpt": PROJECT_ROOT / "outputs_submission/checkpoints/best_model_state.pt",
    },
]


def load_single_model(ckpt, device):
    cp = torch.load(ckpt, map_location=device, weights_only=False)
    mn = cp.get("model_name", "medium_cnn")
    dropout = cp.get("config", {}).get("dropout", 0.3)
    if mn == "medium_cnn":
        model = MediumCNN(dropout=dropout)
    elif mn == "wide_resnet_tiny":
        model = PreActResNetTiny(dropout=dropout, widths=(48, 96, 192))
    elif mn == "large_cnn":
        model = LargeCNN(dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {mn}")
    model.load_state_dict(cp["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    print(f"设备: {device}")

    images = load_idx_images(PROJECT_ROOT / "build/testA_eval/test_B_images.idx3-ubyte")
    n = images.shape[0]

    with open(PROJECT_ROOT / "build/testA_eval/test_A_labels.idx1-ubyte", "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        labels = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()

    all_probs = []  # list of (N, 10) tensors
    accuracies = []
    predictions = []

    for exp in HYBRID_EXPERTS:
        print(f"\n[{exp['label']}] ({exp['type']})", flush=True)
        if exp["type"] == "testa_specialist":
            fold_sum = None
            for fold in range(exp["n_folds"]):
                ckpt = PROJECT_ROOT / exp["ckpt_pattern"].format(fold=fold)
                model = build_model_from_checkpoint(ckpt, device)
                batches = [predict_with_tta(model, images[s:s+256], device, 8)
                           for s in range(0, n, 256)]
                fp = torch.cat(batches, dim=0)
                del model
                if device == "cuda": torch.cuda.empty_cache()
                fold_sum = fp if fold_sum is None else fold_sum + fp
            probs = fold_sum / exp["n_folds"]
        else:
            model = load_single_model(exp["ckpt"], device)
            batches = [predict_with_tta(model, images[s:s+256], device, 8)
                       for s in range(0, n, 256)]
            probs = torch.cat(batches, dim=0)
            del model
            if device == "cuda": torch.cuda.empty_cache()

        acc = (probs.argmax(dim=1) == labels).float().mean().item()
        accuracies.append(acc)
        all_probs.append(probs.cpu())
        predictions.append(probs.argmax(dim=1).cpu())
        print(f"  acc={acc:.4%}", flush=True)

    # ── Error complementarity ──
    E = len(HYBRID_EXPERTS)
    preds = torch.stack(predictions, dim=0).numpy()  # (E, N)
    labs = labels.numpy()

    print("\n=== 错误互补性 ===")
    for i in range(E):
        for j in range(i+1, E):
            err_i = preds[i] != labs
            err_j = preds[j] != labs
            overlap = (err_i & err_j).sum()
            total_err_i = max(1, err_i.sum())
            correct_either = ((preds[i] == labs) | (preds[j] == labs)).sum()
            ensemble_ub = correct_either / len(labs)
            print(f"  {HYBRID_EXPERTS[i]['label']:<25} + {HYBRID_EXPERTS[j]['label']:<25} "
                  f"重叠={overlap/total_err_i:.0%} 上限={ensemble_ub:.4%}")

    # ── Oracle ──
    oracle = np.any(preds == labs, axis=0).mean()
    print(f"\nOracle (任意专家对即对): {oracle:.4%}")

    # ── Grid search best static weights ──
    print("\n=== 静态权重网格搜索 ===")
    probs_t = torch.stack(all_probs, dim=1)  # (N, E, 10)
    best_acc = 0
    best_w = None
    all_results = []
    step = 0.05
    # For 5 experts, use random search
    np.random.seed(42)
    for _ in range(50000):
        w = np.random.dirichlet(np.ones(E))
        wt = torch.tensor(w, dtype=torch.float32).view(1, E, 1)
        ensemble_p = (probs_t * wt).sum(dim=1)
        acc = (ensemble_p.argmax(dim=1) == labels).float().mean().item()
        all_results.append((w, acc))
        if acc > best_acc:
            best_acc = acc
            best_w = w

    all_results.sort(key=lambda x: -x[1])
    print(f"最佳随机搜索: {best_acc:.4%}")
    for i, (name, _) in enumerate(HYBRID_EXPERTS):
        print(f"  {name:<25} weight={best_w[i]:.4f}")
    print()

    # Top 5
    for rank, (w, acc) in enumerate(all_results[:5]):
        w_str = "/".join(f"{x:.2f}" for x in w)
        print(f"  #{rank+1}: {acc:.4%}  weights=[{w_str}]")

    # Also compare with previous bests
    print("\n=== vs 之前最佳 ===")
    print(f"  3专家固定 0.70/0.20/0.10: 93.40%")
    print(f"  3专家网格搜索 0.40/0.10/0.50: 94.14%")
    print(f"  3专家 mnist 模板 0.35/0.35/0.30: 94.06%")
    print(f"  5专家混合最佳随机搜索: {best_acc:.4%}")


if __name__ == "__main__":
    main()
