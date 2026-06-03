"""Build experimental 5-expert dynamic MoE teacher submission package.

Experts:
1. wide_resnet_tiny_raw_seed42, 5 folds
2. medium_anti1_seed2026, 5 folds
3. medium_raw_seed3407, 5 folds
4. robust_v1, single checkpoint
5. MNIST_clean, single checkpoint
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "build" / "dynamic_moe_5expert_submission"
ROOT_PREDICT = PROJECT_ROOT / "predict.py"
SUBMISSION_MODELS = PROJECT_ROOT / "build" / "submission" / "models"
OUTPUTS_SUBMISSION = PROJECT_ROOT / "outputs_submission" / "checkpoints"

MAIN_CODE = r'''
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
'''


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    models_dir = OUT_DIR / "models"
    models_dir.mkdir()

    # Copy 3 kfold experts from existing submission package.
    for name in ["wide_resnet_tiny_raw_seed42", "medium_anti1_seed2026", "medium_raw_seed3407"]:
        shutil.copytree(SUBMISSION_MODELS / name, models_dir / name)

    # Copy single checkpoints.
    (models_dir / "robust_v1").mkdir()
    shutil.copy2(OUTPUTS_SUBMISSION / "robust_expert_best.pt", models_dir / "robust_v1" / "robust_expert_best.pt")
    (models_dir / "MNIST_clean").mkdir()
    shutil.copy2(OUTPUTS_SUBMISSION / "best_model_state.pt", models_dir / "MNIST_clean" / "best_model_state.pt")

    root_text = ROOT_PREDICT.read_text(encoding="utf-8")
    prefix = root_text.split("EXPERTS = [")[0]
    (OUT_DIR / "predict.py").write_text(prefix + MAIN_CODE, encoding="utf-8")

    zip_path = OUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(OUT_DIR.rglob("*")):
            if file.is_file() and file.name != "submission.zip":
                zf.write(file, arcname=str(file.relative_to(OUT_DIR.parent)))
    print(f"created {zip_path} size={zip_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
