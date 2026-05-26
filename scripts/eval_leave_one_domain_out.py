#!/usr/bin/env python
"""
Leave-one-domain-out evaluation script.

For each held-out domain, a new All-domain generalist model is trained on
all OTHER domains, then evaluated on the held-out domain.  This produces a
domain-level breakdown of generalisation (accuracy, class-1 bias, class-8
accuracy, x→1 errors, confusion matrix).

Usage:
  python scripts/eval_leave_one_domain_out.py
      --config experiments/all_domain_<...>.yaml
      --auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data_registry import DomainRegistry, MultiDomainDataset
from src.error_analysis import (
    class_1_bias_summary,
    class_overprediction_ratio,
    per_class_accuracy,
    top_confused_pairs,
    x_to_target_errors,
)
from src.experiment_config import assert_safe_output_dir, load_experiment_config, resolve_experiment_output_dir


def _build_registry(project_root: Path) -> DomainRegistry:
    return DomainRegistry.from_project_root(project_root)


def _load_domain_samples(domain: str, registry: DomainRegistry, evaluate_preprocess: bool) -> list[tuple]:
    """Return list of (image_tensor, label, sample_id) for a domain."""
    raise NotImplementedError(
        "Domain data loading not yet wired; this script provides the evaluation "
        "scaffolding for future all-domain experiments."
    )


def _evaluate_on_domain(model, loader, device: str, use_amp: bool) -> dict[str, Any]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    preds_t = torch.tensor(all_preds)
    labels_t = torch.tensor(all_labels)
    acc = float((preds_t == labels_t).float().mean())
    bias = class_1_bias_summary(preds_t, labels_t)
    return {
        "accuracy": acc,
        "n_samples": len(all_labels),
        "class_1_overprediction_ratio": bias.get("class_1_overprediction_ratio"),
        "class_8_accuracy": per_class_accuracy(preds_t, labels_t, target_class=8),
        "total_x_to_1_errors": x_to_target_errors(preds_t, labels_t).get("total_x_to_target"),
    }


def main():
    parser = argparse.ArgumentParser(description="Leave-one-domain-out evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--auto", action="store_true", help="automatically hold out every registered domain")
    parser.add_argument("--held-out", type=str, default=None, help="single domain to hold out")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("outputs_runs"))
    args = parser.parse_args()

    raw_config = load_experiment_config(args.config)
    project_root = Path(args.project_root).resolve()
    registry = _build_registry(project_root)
    available = registry.list_available()
    print(f"[LODO] registered domains: {available}")

    held_out_domains: list[str] = []
    if args.auto:
        held_out_domains = available
    elif args.held_out:
        held_out_domains = [args.held_out]

    results: dict[str, Any] = {
        "experiment_id": raw_config.get("experiment_id", "lodo"),
        "held_out_domains": held_out_domains,
        "per_domain": {},
    }

    output_base = resolve_experiment_output_dir(raw_config, project_root, args.output_root)
    output_base = output_base / "leave_one_domain_out"
    output_base.mkdir(parents=True, exist_ok=True)

    for held in held_out_domains:
        print(f"\n[LODO] holding out '{held}', training on {[d for d in available if d != held]}")
        results["per_domain"][held] = {
            "status": "scaffold_ready",
            "note": "All-domain training loop not wired yet; this script validated config + domain registry integration.",
        }

    summary_path = output_base / "lodo_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[LODO] summary: {summary_path}")


if __name__ == "__main__":
    main()
