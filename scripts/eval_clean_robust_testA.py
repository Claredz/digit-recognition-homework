"""Evaluate clean and robust models on new TestA."""

import struct, sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from predict import MediumCNN, load_idx_images, predict_with_tta

CLEAN_ROBUST = [
    # (label, checkpoint_path, model_type)
    ("MNIST_clean",        PROJECT_ROOT / "outputs_submission/checkpoints/best_model_state.pt", "medium_cnn"),
    ("robust_v1",          PROJECT_ROOT / "outputs_submission/checkpoints/robust_expert_best.pt", "medium_cnn"),
    ("robust_v2_best",     PROJECT_ROOT / "outputs_submission/checkpoints/robust_expert_v2_best.pt", "medium_cnn"),
    ("robust_v2_long",     PROJECT_ROOT / "outputs_submission/checkpoints/robust_expert_v2_long_best.pt", "medium_cnn"),
    ("robust_v2_testa_pt", PROJECT_ROOT / "outputs_submission/checkpoints/robust_expert_v2_testa_partial_best.pt", "medium_cnn"),
]

# Also add kfold robust models (we'll ensemble them)
KFOLD_ROBUST = [
    PROJECT_ROOT / f"outputs_submission/checkpoints/robust_expert_v2_kfold_f{f}_best.pt"
    for f in range(5)
]


def load_model(ckpt, device):
    cp = torch.load(ckpt, map_location=device, weights_only=False)
    model = MediumCNN(dropout=cp.get("config", {}).get("dropout", 0.3))
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

    results = []

    # Single models
    for label, ckpt, _ in CLEAN_ROBUST:
        model = load_model(ckpt, device)
        batches = [predict_with_tta(model, images[s:s+256], device, 8)
                   for s in range(0, n, 256)]
        probs = torch.cat(batches, dim=0)
        acc = (probs.argmax(dim=1) == labels).float().mean().item()
        conf = probs.max(dim=1).values.mean().item()
        results.append((label, acc))
        print(f"  {label:<20} acc={acc:.4%}  conf={conf:.4f}")
        del model
        if device == "cuda": torch.cuda.empty_cache()

    # KFold robust ensemble
    fold_probs = None
    for f, ckpt in enumerate(KFOLD_ROBUST):
        if not ckpt.exists():
            print(f"  [WARN] kfold_f{f} not found: {ckpt}")
            continue
        model = load_model(ckpt, device)
        batches = [predict_with_tta(model, images[s:s+256], device, 8)
                   for s in range(0, n, 256)]
        fp = torch.cat(batches, dim=0)
        fold_probs = fp if fold_probs is None else fold_probs + fp
        del model
        if device == "cuda": torch.cuda.empty_cache()

    if fold_probs is not None:
        kfold_probs = fold_probs / 5.0
        acc = (kfold_probs.argmax(dim=1) == labels).float().mean().item()
        results.append(("robust_kfold_ens", acc))
        print(f"  robust_kfold_ens     acc={acc:.4%}")

    # Compare with TestA specialist bests (from cached probs)
    cached = torch.load(PROJECT_ROOT / "build/testA_eval/expert_probs.pt", map_location='cpu', weights_only=False)
    specialist_names = ["[S] wide_resnet", "[S] medium_anti1_seed2026", "[S] medium_raw_seed3407"]
    for i, name in enumerate(specialist_names):
        acc = (cached[:, i, :].argmax(dim=1) == labels).float().mean().item()
        results.append((name, acc))

    # Oracle: combine clean MNIST model + best specialist
    mnist_p = torch.cat([predict_with_tta(load_model(CLEAN_ROBUST[0][1], device),
                                           images[s:s+256], device, 8)
                          for s in range(0, n, 256)], dim=0)
    wide_p = cached[:, 0, :]
    # Simple average
    combo = (mnist_p + wide_p) / 2.0
    acc = (combo.argmax(dim=1) == labels).float().mean().item()
    results.append(("MNIST_clean + wide", acc))
    print(f"  MNIST_clean + wide_resnet combo: {acc:.4%}")

    print("\n=== 排序 ===")
    results.sort(key=lambda x: -x[1])
    for name, acc in results:
        bar = "#" * int(acc * 50)
        print(f"  {acc:.4%} {bar} {name}")


if __name__ == "__main__":
    main()
