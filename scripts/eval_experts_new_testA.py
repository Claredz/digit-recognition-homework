"""Evaluate top candidate experts on new TestA — per-expert accuracy and probabilities."""

import struct, sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from predict import (
    MediumCNN, LargeCNN, PreActResNetTiny,
    build_model_from_checkpoint, load_idx_images, predict_with_tta,
)

CANDIDATES = [
    # (run_name, model_name, seed)
    ("testa_wide_resnet_tiny_raw_seed42_e60",             "wide_resnet_tiny",    42),
    ("testa_preact_resnet_tiny_anti1_margin_seed42_e60",  "preact_resnet_tiny",  42),
    ("testa_preact_resnet_tiny_v2_raw_seed42_e60",        "preact_resnet_tiny",  42),
    ("testa_medium_v2_anti1_margin_seed3407_e60",         "medium_cnn",          3407),
    ("testa_medium_v2_raw_seed3407_e60",                  "medium_cnn",          3407),
    ("testa_medium_v2_raw_seed777_e60",                   "medium_cnn",          777),
    ("testa_partial_init_lr1e4_mixup01_erasing005_e40",   "medium_cnn",          42),
    ("testa_large_cnn_v2_raw_seed42_e60",                 "large_cnn",           42),
]


def load_labels():
    with open(PROJECT_ROOT / "build/testA_eval/test_A_labels.idx1-ubyte", "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    print(f"设备: {device}")

    labels = load_labels()
    images = load_idx_images(PROJECT_ROOT / "build/testA_eval/test_B_images.idx3-ubyte")
    n = images.shape[0]

    results = []
    probs_dict = {}

    for run_name, model_name, seed in CANDIDATES:
        run_dir = PROJECT_ROOT / "outputs_runs" / run_name
        fold_probs = None
        missing = 0
        for fold in range(5):
            ckpt = run_dir / f"seed_{seed}" / f"fold_{fold}" / "checkpoints" / "testa_specialist_best.pt"
            if not ckpt.exists():
                # Try auto-discover seed
                alt = list((run_dir / f"seed_{seed}").glob(f"fold_{fold}/checkpoints/testa_specialist_best.pt"))
                if not alt:
                    missing += 1
                    continue
                ckpt = alt[0]

            model = build_model_from_checkpoint(ckpt, device)
            batches = [predict_with_tta(model, images[s:s+256], device, 8)
                       for s in range(0, n, 256)]
            fold_res = torch.cat(batches, dim=0)
            del model
            if device == "cuda": torch.cuda.empty_cache()
            fold_probs = fold_res if fold_probs is None else fold_probs + fold_res

        if missing > 0:
            print(f"  [WARN] {run_name}: {missing}/5 folds missing")
            continue

        expert_probs = fold_probs / 5.0
        preds = expert_probs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        short = run_name.replace("testa_", "").replace("_seed42", "").replace("_e60", "").replace("_e40", "").replace("_e80", "")
        results.append((short, acc))
        probs_dict[run_name] = expert_probs.cpu()
        print(f"  {short:<50} {acc:.4%}")

    # Also include current 3 experts from cache
    cached = torch.load(PROJECT_ROOT / "build/testA_eval/expert_probs.pt", map_location='cpu', weights_only=False)
    for i, name in enumerate(["wide_resnet", "medium_anti1_seed2026", "medium_raw_seed3407"]):
        p = cached[:, i, :]
        acc = (p.argmax(dim=1) == labels).float().mean().item()
        results.append((f"[3exp] {name}", acc))
        print(f"  [3exp] {name:<47} {acc:.4%}")

    print("\n=== 排序 ===")
    results.sort(key=lambda x: -x[1])
    for name, acc in results:
        bar = "#" * int(acc * 50)
        print(f"  {acc:.4%} {bar} {name}")


if __name__ == "__main__":
    main()
