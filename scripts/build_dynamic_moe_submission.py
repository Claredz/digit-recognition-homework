"""Build dynamic MoE teacher submission package."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "build" / "dynamic_moe_submission"
MODELS_SRC = PROJECT_ROOT / "build" / "submission" / "models"
ROOT_PREDICT = PROJECT_ROOT / "predict.py"

MAIN_CODE = r'''
# ---------------------------------------------------------------------------
# 专家与 Router 配置
# ---------------------------------------------------------------------------

EXPERTS = [
    {"name": "wide_resnet_tiny_raw_seed42", "model_name": "wide_resnet_tiny", "static_weight": 0.7, "mnist_weight": 0.35},
    {"name": "medium_anti1_seed2026",       "model_name": "medium_cnn",        "static_weight": 0.2, "mnist_weight": 0.35},
    {"name": "medium_raw_seed3407",         "model_name": "medium_cnn",        "static_weight": 0.1, "mnist_weight": 0.30},
]

DYNAMIC_PARAMS = {"confidence": 0.75, "margin": 0.0, "disagreement": -1.0, "anti1_boost": 0.0}
RULE_PARAMS = {"medium_conf": 0.70, "wide_conf": 0.70, "anti1_delta": 0.15, "wide_slack": 0.10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="动态 MoE 手写数字识别预测")
    parser.add_argument("--testdata", required=True, help="测试数据目录，包含 test_B_images.idx3-ubyte")
    parser.add_argument("--output", required=True, help="预测结果输出路径 (submission.csv)")
    parser.add_argument("--batch-size", type=int, default=256, help="批大小")
    parser.add_argument("--tta-n", type=int, default=8, help="TTA 视图数 (0 或 1 禁用)")
    parser.add_argument("--router", choices=["static", "dynamic", "rule", "mnist"], default="dynamic")
    return parser.parse_args()


def compute_dynamic_weights(probs: torch.Tensor) -> torch.Tensor:
    base = torch.tensor([expert["static_weight"] for expert in EXPERTS], dtype=probs.dtype, device=probs.device)
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]
    mean_distribution = (probs * base.view(1, -1, 1)).sum(dim=1, keepdim=True).clamp_min(1e-9)
    disagreement = (probs.clamp_min(1e-9) * torch.log(probs.clamp_min(1e-9) / mean_distribution)).sum(dim=-1)
    confidence = confidence - confidence.mean(dim=1, keepdim=True)
    margin = margin - margin.mean(dim=1, keepdim=True)
    disagreement = disagreement - disagreement.mean(dim=1, keepdim=True)
    fixed_probs = (probs * base.view(1, -1, 1)).sum(dim=1)
    fixed_pred = fixed_probs.argmax(dim=1)
    anti1_signal = (fixed_pred == 1).float().view(-1, 1) * torch.tensor([[0.0, 1.0, 0.0]], dtype=probs.dtype, device=probs.device)
    p = DYNAMIC_PARAMS
    score = (
        torch.log(base).view(1, -1)
        + p["confidence"] * confidence
        + p["margin"] * margin
        - p["disagreement"] * disagreement
        + p["anti1_boost"] * anti1_signal
    )
    return torch.softmax(score, dim=-1)


def compute_rule_weights(probs: torch.Tensor) -> torch.Tensor:
    templates = torch.tensor(
        [[0.70, 0.20, 0.10], [0.35, 0.35, 0.30], [0.50, 0.30, 0.20], [0.45, 0.45, 0.10]],
        dtype=probs.dtype,
        device=probs.device,
    )
    fixed_probs = (probs * templates[0].view(1, -1, 1)).sum(dim=1)
    fixed_pred = fixed_probs.argmax(dim=1)
    max_probs, expert_preds = probs.max(dim=-1)
    wide_conf = max_probs[:, 0]
    medium_conf = (max_probs[:, 1] + max_probs[:, 2]) * 0.5
    medium_agree = expert_preds[:, 1] == expert_preds[:, 2]
    wide_agrees_medium = (expert_preds[:, 0] == expert_preds[:, 1]) | (expert_preds[:, 0] == expert_preds[:, 2])
    anti1_pred = expert_preds[:, 1]
    anti1_delta = fixed_probs[:, 1] - probs[:, 1, 1]
    mnist_mask = medium_agree & (medium_conf >= RULE_PARAMS["medium_conf"]) & (wide_conf <= medium_conf + RULE_PARAMS["wide_slack"])
    testa_mask = (wide_conf >= RULE_PARAMS["wide_conf"]) & wide_agrees_medium
    anti1_mask = (fixed_pred == 1) & (anti1_pred != 1) & (anti1_delta >= RULE_PARAMS["anti1_delta"])
    template_index = torch.full((probs.shape[0],), 2, dtype=torch.long, device=probs.device)
    template_index = torch.where(testa_mask, torch.zeros_like(template_index), template_index)
    template_index = torch.where(mnist_mask, torch.ones_like(template_index), template_index)
    template_index = torch.where(anti1_mask, torch.full_like(template_index, 3), template_index)
    return templates[template_index]


def router_weights(probs: torch.Tensor, router: str) -> torch.Tensor:
    if router == "dynamic":
        return compute_dynamic_weights(probs)
    if router == "rule":
        return compute_rule_weights(probs)
    key = "mnist_weight" if router == "mnist" else "static_weight"
    return torch.tensor([expert[key] for expert in EXPERTS], dtype=probs.dtype, device=probs.device).view(1, -1).repeat(probs.shape[0], 1)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    idx_path = Path(args.testdata) / "test_B_images.idx3-ubyte"
    if not idx_path.exists():
        print(f"错误：找不到测试图像文件 {idx_path}", file=sys.stderr)
        sys.exit(1)
    images = load_idx_images(idx_path)
    n_samples = images.shape[0]
    root = Path(__file__).resolve().parent
    print(f"router={args.router} device={device} samples={n_samples} tta={args.tta_n}", flush=True)
    print("experts=" + ", ".join(expert["name"] for expert in EXPERTS), flush=True)
    expert_probs = []
    for expert in EXPERTS:
        fold_sum = None
        for fold in range(5):
            ckpt_path = root / "models" / expert["name"] / f"fold_{fold}" / "testa_specialist_best.pt"
            if not ckpt_path.exists():
                print(f"错误：找不到 checkpoint {ckpt_path}", file=sys.stderr)
                sys.exit(1)
            model = build_model_from_checkpoint(ckpt_path, device)
            batches = []
            for start in range(0, n_samples, args.batch_size):
                end = min(start + args.batch_size, n_samples)
                batches.append(predict_with_tta(model, images[start:end], device, args.tta_n))
            fold_probs = torch.cat(batches, dim=0)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            fold_sum = fold_probs if fold_sum is None else fold_sum + fold_probs
        expert_probs.append((fold_sum / 5.0).unsqueeze(1))
    probs = torch.cat(expert_probs, dim=1)
    weights = router_weights(probs, args.router)
    final_probs = (weights.unsqueeze(-1) * probs).sum(dim=1)
    predictions = final_probs.argmax(dim=1).tolist()
    confidences = final_probs.max(dim=1).values.tolist()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(f"{p}\n")
    mean_weights = weights.mean(dim=0).tolist()
    mean_conf = sum(confidences) / max(1, len(confidences))
    print(f"mean_weights={[round(w, 4) for w in mean_weights]}", flush=True)
    print(f"mean_confidence={mean_conf:.4f}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    shutil.copytree(MODELS_SRC, OUT_DIR / "models")
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
