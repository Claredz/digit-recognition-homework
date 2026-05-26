"""Custom losses for the multi-expert TestA system.

This module is opt-in: when no anti-class-1 section is present in the YAML, the
training stack continues to use plain :class:`torch.nn.CrossEntropyLoss` and the
old behavior is exactly preserved.

The class-1 over-prediction observed on TestA (predicted/true ratio ≈ 1.27 in
the current OOF baseline) is a TRAINING-TIME bias, not an inference-time mis-
calibration. Adjusting the softmax temperature or shifting logits at inference
gives only a cosmetic improvement; we still see ~165 samples of {8,9,6,5,3,2,4}
predicted as class 1 with high confidence. Section 三 of the project plan asks
us to inject the correction into the training objective itself.

Design:
    L_total = CE(logits, labels) + lambda * mean_{i: y_i != t} relu(z_t - z_y + m)

    - z_t : logit at the target class (default class 1)
    - z_y : logit at the true label
    - m   : margin (default 0.2)
    - applied only to non-target samples → does NOT suppress true class-1 recall
    - mean over non-target samples → robust to class imbalance in a batch

Mixup compatibility:
    Existing training loops call the criterion twice with the two mixed label
    sets (labels_a, labels_b) and combine the resulting losses linearly with the
    mixup coefficient. Because AntiClass1MarginLoss returns a SCALAR (CE + λ·penalty)
    just like nn.CrossEntropyLoss, this composition is automatic and correct:
    each branch's penalty mask uses its own label set.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiClass1MarginLoss(nn.Module):
    """Cross-entropy + class-1 over-prediction margin penalty.

    Args:
        label_smoothing: forwarded to the underlying CrossEntropyLoss.
        lam:    weight of the margin penalty. Typical range [0.02, 0.15];
                lambda > 0.20 will start to suppress true class-1 recall, watch
                ``class_1_accuracy`` in extended diagnostics if you raise it.
        margin: required gap between true-label logit and target-class logit on
                non-target samples. Typical range [0.10, 0.40].
        target_class: which class the model over-predicts. Default 1 because
                that is the documented TestA failure mode; other classes can be
                used to repurpose this loss for a different bias.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        lam: float = 0.05,
        margin: float = 0.2,
        target_class: int = 1,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.lam = float(lam)
        self.margin = float(margin)
        self.target_class = int(target_class)
        self.base_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        # Detached scalars for optional logging without forcing a GPU sync.
        self.register_buffer("_last_penalty", torch.zeros(()), persistent=False)
        self.register_buffer("_last_ce", torch.zeros(()), persistent=False)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = self.base_criterion(logits, labels)

        mask = labels != self.target_class
        if mask.any():
            target_logit = logits[:, self.target_class]
            true_label_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            per_sample = F.relu(target_logit - true_label_logit + self.margin)
            penalty = (per_sample * mask.to(per_sample.dtype)).sum() / mask.sum().clamp(min=1).to(per_sample.dtype)
        else:
            penalty = logits.new_zeros(())

        loss = ce + self.lam * penalty

        with torch.no_grad():
            self._last_penalty.copy_(penalty.detach())
            self._last_ce.copy_(ce.detach())

        return loss

    @property
    def last_penalty(self) -> float:
        return float(self._last_penalty.item())

    @property
    def last_ce(self) -> float:
        return float(self._last_ce.item())

    def extra_repr(self) -> str:
        return (
            f"label_smoothing={self.label_smoothing}, lam={self.lam}, "
            f"margin={self.margin}, target_class={self.target_class}"
        )


def build_criterion(
    *,
    label_smoothing: float = 0.0,
    anti_class1_loss_config: dict[str, Any] | None = None,
) -> nn.Module:
    """Build the appropriate criterion based on config.

    When ``anti_class1_loss_config`` is None or has ``enabled=False``, returns a
    plain :class:`torch.nn.CrossEntropyLoss` with ``label_smoothing`` set — i.e.
    100% equivalent to the legacy behavior (no surprises for old YAMLs).
    """
    if anti_class1_loss_config and anti_class1_loss_config.get("enabled", False):
        lam = float(anti_class1_loss_config.get("lambda", 0.05))
        margin = float(anti_class1_loss_config.get("margin", 0.2))
        target_class = int(anti_class1_loss_config.get("target_class", 1))
        return AntiClass1MarginLoss(
            label_smoothing=label_smoothing,
            lam=lam,
            margin=margin,
            target_class=target_class,
        )
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


__all__ = ["AntiClass1MarginLoss", "build_criterion"]
