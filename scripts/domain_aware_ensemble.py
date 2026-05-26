#!/usr/bin/env python
"""
Heuristic domain-aware ensemble script.

Loads a set of expert OOF probabilities and uses HeuristicDomainRouter
to produce per-sample weights driven by prediction conflicts and class-1
bias heuristics.  Outputs router-per-sample weights, final predictions,
and diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.ensemble.domain_router import HeuristicDomainRouter, save_router_log
from src.error_analysis import (
    class_1_bias_summary,
    class_overprediction_ratio,
    per_class_accuracy,
    top_confused_pairs,
    x_to_target_errors,
)
from src.experiment_config import resolve_experiment_output_dir


def _load_oof(oof_path: Path) -> dict[str, Any]:
    data = torch.load(oof_path, map_location="cpu")
    return {
        "sample_ids": data["sample_ids"],
        "labels": data["labels"],
        "probabilities": data["probabilities"],
    }


def main():
    parser = argparse.ArgumentParser(description="Heuristic domain-aware ensemble")
    parser.add_argument("--experiments", nargs="+", required=True, help="experiment IDs")
    parser.add_argument("--output-root", type=Path, default=Path("outputs_runs"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--js-threshold", type=float, default=0.3)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    oofs: list[dict[str, Any]] = []
    for exp_id in args.experiments:
        oof_path = project_root / args.output_root / exp_id / "oof" / "oof_probabilities.pt"
        if not oof_path.exists():
            print(f"[domain_ensemble] WARN missing {oof_path}, skipping {exp_id}")
            continue
        oofs.append(_load_oof(oof_path))

    if len(oofs) < 2:
        print("[domain_ensemble] need >=2 experts with valid OOFs")
        return

    ref_ids = oofs[0]["sample_ids"]
    ref_labels = oofs[0]["labels"]

    probs_list: list[torch.Tensor] = []
    for o in oofs:
        assert list(o["sample_ids"]) == list(ref_ids), "sample_ids mismatch across experts"
        probs_list.append(o["probabilities"])

    expert_names = args.experiments
    router = HeuristicDomainRouter(
        expert_pool=expert_names,
        js_threshold=args.js_threshold,
    )
    weights = router.route(probs_list)

    fused = torch.stack(probs_list, dim=0) * weights.T.unsqueeze(0)
    fused_prob = fused.sum(dim=0)
    predictions = fused_prob.argmax(dim=-1)
    labels = torch.tensor(ref_labels)

    acc = float((predictions == labels).float().mean())
    bias = class_1_bias_summary(predictions, labels)
    output_dir = project_root / args.output_root / "domain_aware_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_router_log(output_dir, ref_ids, weights, expert_names, predictions)

    summary: dict[str, Any] = {
        "experts": expert_names,
        "n_experts": len(expert_names),
        "n_samples": len(ref_ids),
        "overall_accuracy": acc,
        "class_1_overprediction_ratio": bias.get("class_1_overprediction_ratio"),
        "class_8_accuracy": per_class_accuracy(predictions, labels, target_class=8),
        "total_x_to_1_errors": x_to_target_errors(predictions, labels).get("total_x_to_target"),
        "top_confused_pairs": top_confused_pairs(predictions, labels, n=10),
    }
    (output_dir / "domain_aware_ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "top_confused_pairs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
