from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def js_divergence(probs_a: torch.Tensor, probs_b: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (probs_a + probs_b)
    kl_a = torch.sum(probs_a * torch.log((probs_a + 1e-9) / (m + 1e-9)), dim=-1)
    kl_b = torch.sum(probs_b * torch.log((probs_b + 1e-9) / (m + 1e-9)), dim=-1)
    return 0.5 * (kl_a + kl_b)


def _safe_list_of_floats(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    return list(value)


class HeuristicDomainRouter:
    """
    Heuristic (not learned) domain-aware router that adjusts per-sample
    expert weights based on prediction conflicts and image statistics.
    """

    def __init__(
        self,
        expert_pool: list[str],
        prior_weights: dict[str, float] | None = None,
        js_threshold: float = 0.3,
        known_num_classes: int = 10,
    ):
        self._pool = list(expert_pool)
        self._n_experts = len(self._pool)
        self._name_to_idx = {name: i for i, name in enumerate(self._pool)}
        self._js_threshold = js_threshold
        self._num_classes = known_num_classes

        default = 1.0 / max(1, self._n_experts)
        if prior_weights is None:
            self._prior = torch.tensor([default] * self._n_experts)
        else:
            self._prior = torch.tensor([prior_weights.get(name, default) for name in self._pool])

    def route(
        self,
        expert_probs: list[torch.Tensor],
    ) -> torch.Tensor:
        n_samples = expert_probs[0].shape[0]
        device = expert_probs[0].device
        weights = self._prior.clone().to(device).unsqueeze(0).expand(n_samples, -1).clone()

        stacked = torch.stack(expert_probs, dim=0)

        for i in range(self._n_experts):
            pi = expert_probs[i]
            top1_i = pi.argmax(dim=-1)
            top2_i = pi.topk(2, dim=-1)[1][:, 1]

            class_1_mask = (top1_i == 1)
            class_1_second_mask = (top2_i == 1)

            conflict_classes = {8, 9, 6, 5, 3, 2, 4}
            anti1_bonus_mask = class_1_mask & torch.isin(
                top2_i, torch.tensor(list(conflict_classes), device=device)
            )
            weights[anti1_bonus_mask | class_1_second_mask, i] += 0.15

        for i in range(self._n_experts):
            for j in range(i + 1, self._n_experts):
                js = js_divergence(expert_probs[i], expert_probs[j])
                high_conflict = js > self._js_threshold
                weights[high_conflict, i] += 0.05
                weights[high_conflict, j] += 0.05

        weights = torch.clamp(weights, min=0.0)
        row_sums = weights.sum(dim=-1, keepdim=True)
        weights = weights / (row_sums + 1e-9)

        return weights


def save_router_log(
    output_dir: Path,
    sample_ids: list[int | str],
    weights: torch.Tensor,
    expert_names: list[str],
    predictions: torch.Tensor,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_np = weights.detach().cpu().numpy()
    preds_np = predictions.detach().cpu().numpy()
    records = []
    for idx, sid in enumerate(sample_ids):
        record = {
            "sample_id": int(sid) if isinstance(sid, (int, np.integer)) else str(sid),
            "prediction": int(preds_np[idx]),
            "weights": {expert_names[i]: float(weights_np[idx, i]) for i in range(len(expert_names))},
        }
        records.append(record)

    with open(output_dir / "router_per_sample.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
