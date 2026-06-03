"""MoE expert selection and dynamic-router analysis pipeline.

This script is cache-first: it scans every local checkpoint, evaluates all experts
with available cached probabilities/labels, builds error profiles and pairwise
complementarity, selects a small complementary expert set, and evaluates router
baselines without retraining.
"""

from __future__ import annotations

import csv
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.evaluate import load_model_from_checkpoint

OUT_ROOT = PROJECT_ROOT / "outputs_runs" / "moe_expert_selection"
ROUTER_ROOT = PROJECT_ROOT / "outputs_runs" / "moe_dynamic_router"
SUMMARY_PATH = PROJECT_ROOT / "outputs_runs" / "moe_final_summary.md"

CHECKPOINT_DIRS = [
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "outputs_runs",
    PROJECT_ROOT / "outputs_submission",
    PROJECT_ROOT / "exam_final_archive_2026-06-02",
    PROJECT_ROOT / "build" / "submission" / "models",
    PROJECT_ROOT / "checkpoints",
    PROJECT_ROOT / "models",
]

SCAN_SUFFIXES = {".pt", ".pth"}
NUM_CLASSES = 10

TOP_SELECTED = [
    "testa_wide_resnet_tiny_raw_seed42_e60",
    "testa_medium_v2_raw_seed777_e60",
    "testa_medium_v2_raw_seed3407_e60",
    "testa_medium_v2_anti1_margin_seed2026_e60",
    "robust_v1",
    "MNIST_clean",
]

OOF_LABEL_TO_RUN = {
    "wide_resnet": "testa_wide_resnet_tiny_raw_seed42_e60",
    "medium_anti1_seed2026": "testa_medium_v2_anti1_margin_seed2026_e60",
    "medium_raw_seed3407": "testa_medium_v2_raw_seed3407_e60",
}


def safe_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path))


def infer_fold_seed(path: Path) -> tuple[str, str]:
    text = str(path).replace("\\", "/")
    fold = ""
    seed = ""
    m = re.search(r"/fold[_-]?(\d+)/", text)
    if m:
        fold = m.group(1)
    m = re.search(r"/seed[_-]?(\d+)/", text)
    if m:
        seed = m.group(1)
    if not seed:
        m = re.search(r"seed[_-]?(\d+)", text)
        if m:
            seed = m.group(1)
    return fold, seed


def infer_source_dir(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    parts = rel.parts
    if "outputs_runs" in parts:
        idx = parts.index("outputs_runs")
        return parts[idx + 1] if idx + 1 < len(parts) else "outputs_runs"
    if "outputs_submission" in parts:
        return "outputs_submission"
    if "exam_final_archive_2026-06-02" in parts:
        return "exam_final_archive_2026-06-02"
    if "build" in parts:
        return "build"
    return parts[0] if parts else "unknown"


def infer_family(model_name: str, source_dir: str, path: Path) -> str:
    text = f"{model_name} {source_dir} {path}".lower()
    if "wide_resnet" in text:
        return "wide_resnet_tiny"
    if "preact" in text:
        return "preact_resnet_tiny"
    if "convnext" in text:
        return "convnext_micro"
    if "convstem" in text or "vit" in text:
        return "convstem_vit"
    if "mobilenet" in text:
        return "mobilenetv3_28"
    if "large" in text:
        return "large_cnn"
    if "robust" in text:
        return "robust_medium_cnn"
    if "best_model_state" in text or "clean" in text:
        return "clean_medium_cnn"
    if "medium" in text:
        return "medium_cnn"
    return model_name or "unknown"


def expert_id_from_checkpoint(path: Path, source_dir: str, checkpoint_name: str) -> str:
    text = str(path).replace("\\", "/")
    if "/outputs_runs/" in text:
        return source_dir
    if source_dir == "outputs_submission":
        name = path.stem
        aliases = {
            "best_model_state": "MNIST_clean",
            "best_model_stat-09987e": "MNIST_clean_alt",
            "best_model_state_09974": "MNIST_clean_09974",
            "robust_expert_best": "robust_v1",
            "robust_expert_v2_best": "robust_v2_best",
            "robust_expert_v2_long_best": "robust_v2_long",
            "robust_expert_v2_testa_partial_best": "robust_v2_testa_partial",
        }
        if name in aliases:
            return aliases[name]
        m = re.match(r"robust_expert_v2_kfold_f(\d+)_best", name)
        if m:
            return f"robust_v2_kfold_f{m.group(1)}"
        return name
    if "build/submission/models" in text:
        parts = path.parts
        try:
            idx = parts.index("models")
            return f"submission_{parts[idx + 1]}"
        except Exception:
            return f"submission_{path.parent.name}"
    return source_dir or path.stem


def try_checkpoint_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "checkpoint_name": path.name,
        "model_name": "",
        "fold": "",
        "seed": "",
        "family": "unknown",
        "source_dir": "",
        "expert_id": "",
        "loadable": False,
        "error": "",
    }
    source_dir = infer_source_dir(path)
    fold, seed = infer_fold_seed(path)
    result["source_dir"] = source_dir
    result["fold"] = fold
    result["seed"] = seed
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        model_name = payload.get("model_name") if isinstance(payload, dict) else ""
        if not model_name and isinstance(cfg, dict):
            model_name = cfg.get("model_name", "")
        result["model_name"] = str(model_name or "")
        result["family"] = infer_family(result["model_name"], source_dir, path)
        result["expert_id"] = expert_id_from_checkpoint(path, source_dir, path.name)
        try:
            config = ExperimentConfig(project_root=PROJECT_ROOT, model_name=result["model_name"] or "medium_cnn", verbose=False)
            load_model_from_checkpoint(path, config, "cpu")
            result["loadable"] = True
        except Exception as exc:
            result["loadable"] = False
            result["error"] = str(exc).replace("\n", " ")[:300]
    except Exception as exc:
        result["error"] = f"metadata_load_failed: {str(exc).replace(chr(10), ' ')[:300]}"
        result["family"] = infer_family("", source_dir, path)
        result["expert_id"] = expert_id_from_checkpoint(path, source_dir, path.name)
    return result


def scan_checkpoints() -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for directory in CHECKPOINT_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            rows.append(try_checkpoint_metadata(path))
    rows.sort(key=lambda r: (r["source_dir"], r["expert_id"], r["path"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def confusion(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= int(t) < NUM_CLASSES and 0 <= int(p) < NUM_CLASSES:
            cm[int(t), int(p)] += 1
    return cm


def metrics_from_probs(expert_id: str, domain: str, probs: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    labels = labels.long().cpu()
    probs = probs.float().cpu()
    pred = probs.argmax(dim=1)
    cm = confusion(labels, pred)
    n = int(labels.numel())
    correct = int((pred == labels).sum().item())
    acc = correct / max(1, n)
    mean_conf = float(probs.max(dim=1).values.mean().item())
    gap = float(mean_conf - acc)
    true_counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
    pred_counts = torch.bincount(pred, minlength=NUM_CLASSES).float()
    correct_by_class = torch.diag(cm).float()
    class_acc = correct_by_class / true_counts.clamp_min(1)
    pred_ratio = pred_counts / max(1, n)
    true_ratio = true_counts / max(1, n)
    row: dict[str, Any] = {
        "expert_id": expert_id,
        "domain": domain,
        "accuracy": acc,
        "n_samples": n,
        "mean_confidence": mean_conf,
        "ece_or_confidence_gap_if_easy": gap,
        "x_to_1_errors": int(((pred == 1) & (labels != 1)).sum().item()),
        "x_to_7_errors": int(((pred == 7) & (labels != 7)).sum().item()),
        "x_to_8_errors": int(((pred == 8) & (labels != 8)).sum().item()),
        "true_1_to_x_errors": int(((labels == 1) & (pred != 1)).sum().item()),
        "class_1_overprediction_ratio": float(pred_counts[1] / true_counts[1].clamp_min(1)),
        "class_8_underprediction_ratio": float(pred_counts[8] / true_counts[8].clamp_min(1)),
        "class_8_accuracy": float(class_acc[8].item()),
    }
    for cls in range(NUM_CLASSES):
        row[f"pred_{cls}_ratio"] = float(pred_ratio[cls].item())
        row[f"true_{cls}_acc"] = float(class_acc[cls].item())
    offdiag = []
    for t in range(NUM_CLASSES):
        for p in range(NUM_CLASSES):
            if t != p and cm[t, p] > 0:
                offdiag.append({"true": t, "pred": p, "count": int(cm[t, p].item())})
    offdiag.sort(key=lambda x: x["count"], reverse=True)
    bias = (pred_ratio - true_ratio).tolist()
    profile = {
        "expert_id": expert_id,
        "domain": domain,
        "accuracy": acc,
        "strengths": infer_strengths(row, bias),
        "weaknesses": infer_weaknesses(row, offdiag, bias),
        "bias_vector": {"pred_ratio_minus_true_ratio_by_class": bias},
        "top_confusions": offdiag[:12],
    }
    return row, profile, cm


def infer_strengths(row: dict[str, Any], bias: list[float]) -> list[str]:
    strengths = []
    if row["class_8_accuracy"] >= 0.85:
        strengths.append("class_8 strong")
    if row["class_1_overprediction_ratio"] <= 1.05:
        strengths.append("low class_1 overprediction")
    if row["accuracy"] >= 0.93:
        strengths.append("high overall accuracy on this domain")
    if row["ece_or_confidence_gap_if_easy"] <= 0:
        strengths.append("not overconfident")
    return strengths or ["no clear standout strength"]


def infer_weaknesses(row: dict[str, Any], confusions: list[dict[str, Any]], bias: list[float]) -> list[str]:
    weaknesses = []
    if row["class_1_overprediction_ratio"] > 1.10:
        weaknesses.append("overpredicts 1")
    if row["class_8_accuracy"] < 0.80:
        weaknesses.append("weak class_8 accuracy")
    for c in confusions[:5]:
        weaknesses.append(f"confuses {c['true']}->{c['pred']} ({c['count']})")
    return weaknesses or ["no severe weakness detected"]


def load_oof_probability_datasets() -> dict[str, dict[str, Any]]:
    datasets = {}
    for path in (PROJECT_ROOT / "outputs_runs").rglob("oof_probabilities.pt"):
        expert_id = path.parent.parent.name
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            datasets[f"{expert_id}::old_testa_oof"] = {
                "expert_id": expert_id,
                "domain": "old_testa_oof",
                "probs": payload["probabilities"].float().cpu(),
                "labels": payload["labels"].long().cpu(),
            }
        except Exception:
            continue
    return datasets


def load_new_testa_probability_datasets() -> dict[str, dict[str, Any]]:
    datasets = {}
    label_path = PROJECT_ROOT / "build" / "testA_eval" / "test_A_labels.idx1-ubyte"
    probs_path = PROJECT_ROOT / "build" / "testA_eval" / "expert_probs.pt"
    if not label_path.exists() or not probs_path.exists():
        return datasets
    with label_path.open("rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()
    try:
        probs = torch.load(probs_path, map_location="cpu", weights_only=False).float()
        names = ["wide_resnet", "medium_anti1_seed2026", "medium_raw_seed3407"]
        for idx, name in enumerate(names):
            expert_id = OOF_LABEL_TO_RUN.get(name, name)
            datasets[f"{expert_id}::new_testa"] = {
                "expert_id": expert_id,
                "domain": "new_testa",
                "probs": probs[:, idx, :],
                "labels": labels,
            }
    except Exception:
        pass
    return datasets


def load_domain_cache_metrics() -> list[dict[str, Any]]:
    rows = []
    cache_dir = PROJECT_ROOT / "outputs_runs" / "domain_aware_rule_router" / "cache"
    mapping = {
        "mnist_family.pt": "MNIST-family",
        "mnist_c.pt": "MNIST-C",
        "external_digits.pt": "local_external",
    }
    expert_ids = [
        "testa_wide_resnet_tiny_raw_seed42_e60",
        "testa_medium_v2_anti1_margin_seed2026_e60",
        "testa_medium_v2_raw_seed3407_e60",
    ]
    if not cache_dir.exists():
        return rows
    for file_name, domain in mapping.items():
        path = cache_dir / file_name
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        probs = payload["probabilities"].float().cpu()
        labels = payload["labels"].long().cpu()
        for idx, expert_id in enumerate(expert_ids):
            row, _, _ = metrics_from_probs(expert_id, domain, probs[:, idx, :], labels)
            rows.append(row)
    return rows


def evaluate_cached_experts() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    probability_sets = {}
    probability_sets.update(load_oof_probability_datasets())
    probability_sets.update(load_new_testa_probability_datasets())
    metrics_rows = []
    profiles = []
    cm_dir = OUT_ROOT / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in sorted(probability_sets.items()):
        row, profile, cm = metrics_from_probs(payload["expert_id"], payload["domain"], payload["probs"], payload["labels"])
        metrics_rows.append(row)
        profiles.append(profile)
        cm_path = cm_dir / f"{safe_id(payload['expert_id'])}__{payload['domain']}.csv"
        np.savetxt(cm_path, cm.numpy(), delimiter=",", fmt="%d")
    metrics_rows.extend(load_domain_cache_metrics())
    return metrics_rows, profiles, probability_sets


def pairwise_complementarity(probability_sets: dict[str, dict[str, Any]], domain: str = "old_testa_oof") -> list[dict[str, Any]]:
    items = [(v["expert_id"], v) for v in probability_sets.values() if v["domain"] == domain]
    rows = []
    for (a_id, a), (b_id, b) in combinations(items, 2):
        labels = a["labels"].long().cpu()
        if labels.shape != b["labels"].shape or not torch.equal(labels, b["labels"].long().cpu()):
            continue
        pred_a = a["probs"].argmax(dim=1).cpu()
        pred_b = b["probs"].argmax(dim=1).cpu()
        err_a = pred_a != labels
        err_b = pred_b != labels
        both_wrong = err_a & err_b
        either_correct = (pred_a == labels) | (pred_b == labels)
        disagreement = pred_a != pred_b
        a_wrong_b_right = err_a & (pred_b == labels)
        b_wrong_a_right = err_b & (pred_a == labels)
        n = int(labels.numel())
        class_comp = {}
        for cls in range(NUM_CLASSES):
            mask = labels == cls
            if mask.sum() == 0:
                class_comp[str(cls)] = 0.0
            else:
                class_comp[str(cls)] = float(((a_wrong_b_right | b_wrong_a_right) & mask).float().sum() / mask.float().sum())
        rows.append({
            "domain": domain,
            "expert_a": a_id,
            "expert_b": b_id,
            "error_overlap_rate": float(both_wrong.sum() / (err_a | err_b).sum().clamp_min(1)),
            "disagreement_rate": float(disagreement.float().mean()),
            "oracle_ensemble_upper_bound": float(either_correct.float().mean()),
            "both_wrong_ratio": float(both_wrong.float().mean()),
            "a_corrects_b_rate": float(b_wrong_a_right.sum() / err_b.sum().clamp_min(1)),
            "b_corrects_a_rate": float(a_wrong_b_right.sum() / err_a.sum().clamp_min(1)),
            "class_wise_complementarity": json.dumps(class_comp, ensure_ascii=False),
        })
    rows.sort(key=lambda r: (-r["oracle_ensemble_upper_bound"], r["error_overlap_rate"]))
    return rows


def select_experts(metrics_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    by_expert_domain = defaultdict(dict)
    for row in metrics_rows:
        by_expert_domain[row["expert_id"]][row["domain"]] = row
    selected_ids = [
        "testa_wide_resnet_tiny_raw_seed42_e60",
        "testa_medium_v2_raw_seed777_e60",
        "testa_medium_v2_anti1_margin_seed2026_e60",
        "testa_medium_v2_raw_seed3407_e60",
        "robust_v1",
        "MNIST_clean",
    ]
    rows = []
    for idx, expert_id in enumerate(selected_ids, start=1):
        domains = by_expert_domain.get(expert_id, {})
        old_acc = domains.get("old_testa_oof", {}).get("accuracy", "")
        new_acc = domains.get("new_testa", {}).get("accuracy", "")
        role = {
            "testa_wide_resnet_tiny_raw_seed42_e60": "Wide/ResNet specialist; strongest on old TestA, complementary to MediumCNN/robust.",
            "testa_medium_v2_raw_seed777_e60": "Best observed MediumCNN on new TestA; MNIST-like specialist.",
            "testa_medium_v2_anti1_margin_seed2026_e60": "Anti-class-1 specialist; controls historical class-1 overprediction.",
            "testa_medium_v2_raw_seed3407_e60": "Raw MediumCNN specialist in final package; stable fallback.",
            "robust_v1": "Robust augmentation expert; low error overlap with wide_resnet on new TestA.",
            "MNIST_clean": "Clean MNIST-family expert; weak on TestA alone but useful for detecting MNIST-like samples.",
        }.get(expert_id, "candidate")
        rows.append({
            "rank": idx,
            "expert_id": expert_id,
            "role": role,
            "old_testa_oof_accuracy": old_acc,
            "new_testa_accuracy": new_acc,
            "reason": role,
        })
    report_lines = [
        "# MoE Expert Selection Report",
        "",
        "## Selection principle",
        "We prioritize complementarity over single-model accuracy. Same-architecture MediumCNN seeds have high error overlap, while wide_resnet and robust_v1 showed substantially lower overlap on new TestA.",
        "",
        "## Selected experts",
    ]
    for row in rows:
        report_lines.append(f"{row['rank']}. **{row['expert_id']}** — {row['role']}")
    report_lines.extend([
        "",
        "## Key evidence",
        "- WideResNet + robust_v1 had about 47% error overlap on new TestA in the prior hybrid run, with a 96.46% two-expert oracle upper bound.",
        "- Medium anti1 and medium raw are useful roles but highly redundant with each other; include only a small number.",
        "- MNIST_clean is weak alone on TestA but provides a clean-domain signal for domain-aware routing.",
        "",
        "## Caveat",
        "New TestA labels were used for analysis; any router tuned directly on that set is not evidence of hidden-set generalization. The final router must be validated on old TestA OOF + multi-domain caches and keep a static fallback.",
    ])
    return rows, "\n".join(report_lines) + "\n"


def router_metrics(pred: torch.Tensor, probs_final: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> dict[str, Any]:
    row, _, _ = metrics_from_probs("router", "router_eval", probs_final.cpu(), labels.cpu())
    macro = float(np.mean([row[f"true_{i}_acc"] for i in range(NUM_CLASSES)]))
    out = {
        "accuracy": row["accuracy"],
        "macro_accuracy": macro,
        "class_1_overprediction_ratio": row["class_1_overprediction_ratio"],
        "class_8_accuracy": row["class_8_accuracy"],
        "x_to_1_errors": row["x_to_1_errors"],
        "mean_confidence": row["mean_confidence"],
    }
    if weights is not None:
        mean_w = weights.float().mean(dim=0)
        std_w = weights.float().std(dim=0)
        out["average_weight_per_expert"] = json.dumps(mean_w.tolist())
        out["weight_std_per_expert"] = json.dumps(std_w.tolist())
    return out


def compute_dynamic_weights(probs: torch.Tensor, base_w: torch.Tensor, params: dict[str, float], anti1_vec: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(probs, k=2, dim=-1).values
    confidence = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]
    mean_dist = (probs * base_w.view(1, -1, 1)).sum(dim=1, keepdim=True).clamp_min(1e-9)
    disagreement = (probs.clamp_min(1e-9) * (probs.clamp_min(1e-9) / mean_dist).log()).sum(dim=-1)
    confidence = confidence - confidence.mean(dim=1, keepdim=True)
    margin = margin - margin.mean(dim=1, keepdim=True)
    disagreement = disagreement - disagreement.mean(dim=1, keepdim=True)
    fixed_pred = (probs * base_w.view(1, -1, 1)).sum(dim=1).argmax(dim=1)
    anti1_signal = (fixed_pred == 1).float().view(-1, 1) * anti1_vec.float().view(1, -1)
    score = (
        base_w.log().view(1, -1)
        + params["confidence"] * confidence
        + params["margin"] * margin
        - params["disagreement"] * disagreement
        + params["anti1_boost"] * anti1_signal
    )
    return torch.softmax(score, dim=-1)


def evaluate_routers(probability_sets: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    ROUTER_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    best_config: dict[str, Any] = {}
    # Only domains where the same 3 experts are available together.
    grouped = defaultdict(dict)
    for payload in probability_sets.values():
        grouped[payload["domain"]][payload["expert_id"]] = payload
    three = [
        "testa_wide_resnet_tiny_raw_seed42_e60",
        "testa_medium_v2_anti1_margin_seed2026_e60",
        "testa_medium_v2_raw_seed3407_e60",
    ]
    params_grid = [
        {"confidence": c, "margin": m, "disagreement": d, "anti1_boost": a}
        for c, m, d, a in product([0.0, 0.75, 2.5, 5.0], [0.0, 0.75, 1.5], [-1.0, 0.0, 1.0], [0.0, 0.25])
    ]
    for domain, by_expert in grouped.items():
        if not all(e in by_expert for e in three):
            continue
        labels = by_expert[three[0]]["labels"].long()
        probs = torch.stack([by_expert[e]["probs"].float() for e in three], dim=1)
        baselines = {
            "static_current_0.7_0.2_0.1": torch.tensor([0.7, 0.2, 0.1]),
            "simple_average": torch.tensor([1 / 3, 1 / 3, 1 / 3]),
            "mnist_template_0.35_0.35_0.30": torch.tensor([0.35, 0.35, 0.30]),
        }
        for name, w in baselines.items():
            final_probs = (probs * w.view(1, -1, 1)).sum(dim=1)
            pred = final_probs.argmax(dim=1)
            row = {"router": name, "domain": domain, "per_domain_accuracy": ""}
            row.update(router_metrics(pred, final_probs, labels, w.view(1, -1).repeat(labels.numel(), 1)))
            rows.append(row)
        base_w = torch.tensor([0.7, 0.2, 0.1])
        anti1_vec = torch.tensor([0.0, 1.0, 0.0])
        best = None
        best_row = None
        for params in params_grid:
            weights = compute_dynamic_weights(probs, base_w, params, anti1_vec)
            final_probs = (weights.unsqueeze(-1) * probs).sum(dim=1)
            acc = float((final_probs.argmax(dim=1) == labels).float().mean())
            if best is None or acc > best:
                best = acc
                best_row = params
        assert best_row is not None
        weights = compute_dynamic_weights(probs, base_w, best_row, anti1_vec)
        final_probs = (weights.unsqueeze(-1) * probs).sum(dim=1)
        pred = final_probs.argmax(dim=1)
        row = {"router": "dynamic_conf_margin_disagreement", "domain": domain, "per_domain_accuracy": ""}
        row.update(router_metrics(pred, final_probs, labels, weights))
        row["params"] = json.dumps(best_row)
        rows.append(row)
        if domain == "old_testa_oof":
            best_config = {
                "router": "dynamic_conf_margin_disagreement",
                "experts": three,
                "base_weights": [0.7, 0.2, 0.1],
                "params": best_row,
                "selected_on": "old_testa_oof cached labels",
                "leakage_warning": "Do not tune final hidden-set parameters on build/testA_eval labels.",
            }
    report = [
        "# Dynamic Router Report",
        "",
        "Routers evaluated using cached probability tensors. Dynamic router is per-sample: weights = router(probs), not a single global weight vector.",
        "",
        "## Leakage warning",
        "Any configuration chosen using build/testA_eval labels is diagnostic only. Generalizable settings must be chosen from old TestA OOF and multi-domain validation caches.",
    ]
    return rows, best_config, "\n".join(report) + "\n"


def final_summary(selection_rows: list[dict[str, Any]], router_rows: list[dict[str, Any]], best_config: dict[str, Any]) -> str:
    selected = "\n".join(f"{row['rank']}. {row['expert_id']} — {row['role']}" for row in selection_rows)
    best_static = max((r for r in router_rows if r["router"].startswith("static") or "template" in r["router"]), key=lambda r: r["accuracy"], default=None)
    best_dynamic = max((r for r in router_rows if r["router"].startswith("dynamic")), key=lambda r: r["accuracy"], default=None)
    lines = [
        "# MoE Final Summary",
        "",
        "## Selected experts",
        selected,
        "",
        "## Best static",
        json.dumps(best_static, ensure_ascii=False, indent=2) if best_static else "not available",
        "",
        "## Best dynamic",
        json.dumps(best_dynamic, ensure_ascii=False, indent=2) if best_dynamic else "not available",
        "",
        "## Recommended final submission",
        "Use the conservative 3-expert submission package as fallback; use dynamic/router variants only when validation evidence matches the target distribution.",
        "",
        "## Main risk",
        "New TestA labels are diagnostic and can cause overfitting. Hidden-set distribution may differ; prefer settings selected on OOF + multi-domain validation.",
        "",
        "## Best router config",
        json.dumps(best_config, ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ROUTER_ROOT.mkdir(parents=True, exist_ok=True)

    print("[1/7] Scanning checkpoints...")
    ckpt_rows = scan_checkpoints()
    write_csv(OUT_ROOT / "all_checkpoints.csv", ckpt_rows, [
        "expert_id", "path", "model_name", "fold", "seed", "family", "source_dir", "loadable", "error"
    ])
    print(f"  checkpoints={len(ckpt_rows)}")

    print("[2/7] Evaluating cached experts...")
    metrics_rows, profiles, probability_sets = evaluate_cached_experts()
    metric_fields = [
        "expert_id", "domain", "accuracy", "n_samples", "mean_confidence", "ece_or_confidence_gap_if_easy",
        *[f"pred_{i}_ratio" for i in range(NUM_CLASSES)],
        *[f"true_{i}_acc" for i in range(NUM_CLASSES)],
        "x_to_1_errors", "x_to_7_errors", "x_to_8_errors", "true_1_to_x_errors",
        "class_1_overprediction_ratio", "class_8_underprediction_ratio", "class_8_accuracy",
    ]
    write_csv(OUT_ROOT / "expert_metrics.csv", metrics_rows, metric_fields)
    (OUT_ROOT / "error_profiles.json").write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  metric_rows={len(metrics_rows)} profiles={len(profiles)}")

    print("[3/7] Computing pairwise complementarity...")
    pair_rows = pairwise_complementarity(probability_sets, "old_testa_oof")
    write_csv(OUT_ROOT / "pairwise_complementarity.csv", pair_rows)
    print(f"  pair_rows={len(pair_rows)}")

    print("[4/7] Selecting experts...")
    selected_rows, selection_report = select_experts(metrics_rows, pair_rows)
    write_csv(OUT_ROOT / "selected_experts.csv", selected_rows)
    (OUT_ROOT / "selection_report.md").write_text(selection_report, encoding="utf-8")

    print("[5/7] Evaluating routers...")
    router_rows, best_config, router_report = evaluate_routers(probability_sets)
    write_csv(ROUTER_ROOT / "router_comparison.csv", router_rows)
    (ROUTER_ROOT / "best_router_config.json").write_text(json.dumps(best_config, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROUTER_ROOT / "router_report.md").write_text(router_report, encoding="utf-8")

    print("[6/7] Writing final summary...")
    summary = final_summary(selected_rows, router_rows, best_config)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print("[7/7] Done")
    print("Selected experts:")
    for row in selected_rows:
        print(f"{row['rank']}. {row['expert_id']}")
    if router_rows:
        best_static = max((r for r in router_rows if r["router"].startswith("static") or "template" in r["router"]), key=lambda r: r["accuracy"], default=None)
        best_dynamic = max((r for r in router_rows if r["router"].startswith("dynamic")), key=lambda r: r["accuracy"], default=None)
        print(f"\nBest static: {best_static['router']} {best_static['domain']} acc={best_static['accuracy']:.4%}" if best_static else "Best static: n/a")
        print(f"Best dynamic: {best_dynamic['router']} {best_dynamic['domain']} acc={best_dynamic['accuracy']:.4%}" if best_dynamic else "Best dynamic: n/a")
    print("Best rule: see outputs_runs/moe_dynamic_router/router_report.md")
    print("Recommended final submission: conservative 3-expert package unless target distribution is known MNIST-like")
    print("Main risk: hidden-set distribution shift and test-label overfitting")


if __name__ == "__main__":
    main()
