"""Train and cross-validate a learned MoE router on cached expert probabilities.

The router is intentionally small (linear gate or tiny MLP). It uses cached
[N, 3, 10] expert probabilities from domain-aware rule-router evaluation, so it
never reruns the CNN/ResNet experts during router training.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERTS = [
    {
        "label": "wide_resnet_tiny_raw_seed42",
        "base_weight": 0.7,
        "oof": PROJECT_ROOT / "outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_anti1_seed2026",
        "base_weight": 0.2,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_raw_seed3407",
        "base_weight": 0.1,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_raw_seed3407_e60/oof/oof_probabilities.pt",
    },
]

DOMAIN_WEIGHTS = {
    "hidden_b_balanced": {"TestA": 0.45, "MNIST-family": 0.35, "MNIST-C": 0.20},
    "hidden_b_easy": {"TestA": 0.35, "MNIST-family": 0.45, "MNIST-C": 0.20},
    "hidden_b_hard": {"TestA": 0.70, "MNIST-family": 0.15, "MNIST-C": 0.15},
}

CACHE_DIR = PROJECT_ROOT / "outputs_runs/domain_aware_rule_router/cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs_runs/learned_moe_router"
TRAIN_DOMAINS = ["TestA", "MNIST-family", "MNIST-C"]
DIAGNOSTIC_DOMAINS = ["local/external digits"]


@dataclass(frozen=True)
class TrainConfig:
    model_type: str
    objective: str
    kl_coef: float
    entropy_coef: float
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 220
    patience: int = 35


CONFIGS = [
    TrainConfig("linear", "hidden_b_balanced", 0.00, 0.00),
    TrainConfig("linear", "hidden_b_balanced", 0.02, 0.00),
    TrainConfig("linear", "hidden_b_balanced", 0.05, 0.00),
    TrainConfig("linear", "hidden_b_easy", 0.02, 0.00),
    TrainConfig("linear", "hidden_b_hard", 0.02, 0.00),
    TrainConfig("mlp", "hidden_b_balanced", 0.02, 0.00, lr=2e-3, weight_decay=2e-4, epochs=260, patience=40),
    TrainConfig("mlp", "hidden_b_easy", 0.02, 0.00, lr=2e-3, weight_decay=2e-4, epochs=260, patience=40),
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_testa_oof() -> dict[str, torch.Tensor]:
    probabilities = []
    labels = None
    sample_ids = None
    for expert in EXPERTS:
        payload = torch.load(expert["oof"], map_location="cpu")
        if labels is None:
            labels = payload["labels"].long()
            sample_ids = payload["sample_ids"]
        elif sample_ids != payload["sample_ids"] or not torch.equal(labels, payload["labels"].long()):
            raise ValueError(f"OOF sample order mismatch: {expert['label']}")
        probabilities.append(payload["probabilities"].float())
    return {"probabilities": torch.stack(probabilities, dim=1), "labels": labels}


def load_domain_payloads() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "TestA": load_testa_oof(),
        "MNIST-family": torch.load(CACHE_DIR / "mnist_family.pt", map_location="cpu"),
        "MNIST-C": torch.load(CACHE_DIR / "mnist_c.pt", map_location="cpu"),
        "local/external digits": torch.load(CACHE_DIR / "external_digits.pt", map_location="cpu"),
    }


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=-1)


def js_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (a + b)
    return 0.5 * (
        (a.clamp_min(1e-9) * (a.clamp_min(1e-9) / m.clamp_min(1e-9)).log()).sum(dim=-1)
        + (b.clamp_min(1e-9) * (b.clamp_min(1e-9) / m.clamp_min(1e-9)).log()).sum(dim=-1)
    )


def make_features(probs: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(probs, k=2, dim=-1).values
    max_probs = top2[..., 0]
    margins = top2[..., 0] - top2[..., 1]
    entropies = entropy(probs)
    preds = probs.argmax(dim=-1)
    fixed = torch.einsum("nec,e->nc", probs, torch.tensor([0.7, 0.2, 0.1], dtype=probs.dtype))
    fixed_pred = fixed.argmax(dim=-1)
    fixed_top2 = torch.topk(fixed, k=2, dim=-1).values

    agreement_features = torch.stack(
        [
            (preds[:, 0] == preds[:, 1]).float(),
            (preds[:, 0] == preds[:, 2]).float(),
            (preds[:, 1] == preds[:, 2]).float(),
            (preds[:, 0] == fixed_pred).float(),
            (preds[:, 1] == fixed_pred).float(),
            (preds[:, 2] == fixed_pred).float(),
            (fixed_pred == 1).float(),
            ((fixed_pred == 1) & (preds[:, 1] != 1)).float(),
        ],
        dim=1,
    )
    js_features = torch.stack(
        [
            js_divergence(probs[:, 0], probs[:, 1]),
            js_divergence(probs[:, 0], probs[:, 2]),
            js_divergence(probs[:, 1], probs[:, 2]),
        ],
        dim=1,
    )
    fixed_features = torch.cat([fixed, fixed_top2[:, :1], (fixed_top2[:, :1] - fixed_top2[:, 1:2])], dim=1)
    return torch.cat(
        [
            probs.flatten(start_dim=1),
            max_probs,
            margins,
            entropies,
            js_features,
            agreement_features,
            fixed_features,
        ],
        dim=1,
    )


class FeatureStandardizer:
    def __init__(self):
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, features: torch.Tensor):
        self.mean = features.mean(dim=0, keepdim=True)
        self.std = features.std(dim=0, keepdim=True).clamp_min(1e-6)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("standardizer is not fitted")
        return (features - self.mean.to(features.device)) / self.std.to(features.device)


class LinearRouter(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class TinyMlpRouter(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 24), nn.ReLU(), nn.Dropout(0.08), nn.Linear(24, 3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_model(model_type: str, input_dim: int) -> nn.Module:
    if model_type == "linear":
        return LinearRouter(input_dim)
    if model_type == "mlp":
        return TinyMlpRouter(input_dim)
    raise ValueError(f"unknown model_type={model_type}")


def final_probs_from_router(model: nn.Module, features: torch.Tensor, expert_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(model(features), dim=-1)
    final_probs = torch.sum(weights.unsqueeze(-1) * expert_probs, dim=1).clamp_min(1e-9)
    return final_probs, weights


def weighted_router_loss(
    model: nn.Module,
    features: torch.Tensor,
    expert_probs: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
    prior_weights: torch.Tensor,
    kl_coef: float,
    entropy_coef: float,
) -> torch.Tensor:
    final_probs, router_weights = final_probs_from_router(model, features, expert_probs)
    nll = -torch.log(final_probs[torch.arange(labels.numel(), device=labels.device), labels])
    loss = (nll * sample_weights).sum() / sample_weights.sum().clamp_min(1e-9)
    if kl_coef > 0:
        kl = torch.sum(router_weights * (router_weights.clamp_min(1e-9).log() - prior_weights.log()), dim=1)
        loss = loss + kl_coef * (kl * sample_weights).sum() / sample_weights.sum().clamp_min(1e-9)
    if entropy_coef > 0:
        router_entropy = -torch.sum(router_weights * router_weights.clamp_min(1e-9).log(), dim=1)
        loss = loss - entropy_coef * (router_entropy * sample_weights).sum() / sample_weights.sum().clamp_min(1e-9)
    return loss


def accuracy(model: nn.Module, features: torch.Tensor, expert_probs: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        probs, _ = final_probs_from_router(model, features, expert_probs)
        preds = probs.argmax(dim=1)
        return float((preds == labels).float().mean().item())


def domain_metrics(model: nn.Module, domain_data: dict, device: str, standardizer: FeatureStandardizer) -> dict:
    features = standardizer.transform(domain_data["features"].to(device))
    probs = domain_data["probabilities"].to(device)
    labels = domain_data["labels"].to(device)
    with torch.no_grad():
        final_probs, router_weights = final_probs_from_router(model, features, probs)
        preds = final_probs.argmax(dim=1)
    return {
        "accuracy": float((preds == labels).float().mean().item()),
        "num_samples": int(labels.numel()),
        "mean_weights": router_weights.mean(dim=0).detach().cpu().tolist(),
    }


def objective_score(metrics: dict, objective_name: str) -> float:
    return sum(metrics[domain]["accuracy"] * weight for domain, weight in DOMAIN_WEIGHTS[objective_name].items())


def train_one_fold(
    config: TrainConfig,
    fold: int,
    train_parts: list[dict],
    val_parts: list[dict],
    all_domains: dict,
    device: str,
) -> dict:
    train_features_cpu = torch.cat([part["features"] for part in train_parts], dim=0)
    standardizer = FeatureStandardizer()
    standardizer.fit(train_features_cpu)

    train_features = standardizer.transform(train_features_cpu).to(device)
    train_probs = torch.cat([part["probabilities"] for part in train_parts], dim=0).to(device)
    train_labels = torch.cat([part["labels"] for part in train_parts], dim=0).to(device)
    train_weights = torch.cat([part["sample_weights"] for part in train_parts], dim=0).to(device)

    val_features = standardizer.transform(torch.cat([part["features"] for part in val_parts], dim=0)).to(device)
    val_probs = torch.cat([part["probabilities"] for part in val_parts], dim=0).to(device)
    val_labels = torch.cat([part["labels"] for part in val_parts], dim=0).to(device)

    model = build_model(config.model_type, train_features.shape[1]).to(device)
    prior_weights = torch.tensor([e["base_weight"] for e in EXPERTS], device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_val = -1.0
    bad_epochs = 0
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = weighted_router_loss(
            model,
            train_features,
            train_probs,
            train_labels,
            train_weights,
            prior_weights,
            config.kl_coef,
            config.entropy_coef,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        val_acc = accuracy(model, val_features, val_probs, val_labels)
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    fold_metrics = {}
    for domain_name, domain_data in all_domains.items():
        fold_metrics[domain_name] = domain_metrics(model, domain_data, device, standardizer)
    fold_metrics["objective_score"] = objective_score(fold_metrics, config.objective)
    return {
        "fold": fold,
        "best_val_accuracy_on_combined_holdout": best_val,
        "epochs_ran": epoch + 1,
        "metrics": fold_metrics,
    }


def make_cv_parts(domains: dict, objective_name: str, fold: int, n_splits: int = 5):
    train_parts = []
    val_parts = []
    rng_seed = 42 + fold
    for domain_name in TRAIN_DOMAINS:
        domain = domains[domain_name]
        labels_np = domain["labels"].numpy()
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(np.zeros_like(labels_np), labels_np))
        train_idx_np, val_idx_np = splits[fold]
        train_idx = torch.tensor(train_idx_np, dtype=torch.long)
        val_idx = torch.tensor(val_idx_np, dtype=torch.long)
        domain_weight = DOMAIN_WEIGHTS[objective_name][domain_name]
        sample_weight = torch.full((len(train_idx),), domain_weight / max(1, len(train_idx)), dtype=torch.float32)
        train_parts.append(
            {
                "features": domain["features"][train_idx],
                "probabilities": domain["probabilities"][train_idx],
                "labels": domain["labels"][train_idx],
                "sample_weights": sample_weight,
            }
        )
        val_parts.append(
            {
                "features": domain["features"][val_idx],
                "probabilities": domain["probabilities"][val_idx],
                "labels": domain["labels"][val_idx],
            }
        )
    return train_parts, val_parts


def run_config(config: TrainConfig, domains: dict, device: str) -> dict:
    set_seed(1234)
    folds = []
    for fold in range(5):
        train_parts, val_parts = make_cv_parts(domains, config.objective, fold)
        folds.append(train_one_fold(config, fold, train_parts, val_parts, domains, device))
        print(
            f"[{config.model_type}/{config.objective}/kl={config.kl_coef}] fold={fold} "
            f"objective={folds[-1]['metrics']['objective_score']:.6f}",
            flush=True,
        )
    mean_metrics = {}
    for domain_name in domains:
        mean_metrics[domain_name] = {
            "accuracy": float(np.mean([fold["metrics"][domain_name]["accuracy"] for fold in folds])),
            "mean_weights": np.mean([fold["metrics"][domain_name]["mean_weights"] for fold in folds], axis=0).tolist(),
            "num_samples": int(domains[domain_name]["labels"].numel()),
        }
    mean_objective = float(np.mean([fold["metrics"]["objective_score"] for fold in folds]))
    return {"config": config.__dict__, "mean_objective_score": mean_objective, "mean_metrics": mean_metrics, "folds": folds}


def static_metrics(domains: dict, weights: list[float]) -> dict:
    result = {}
    w = torch.tensor(weights, dtype=torch.float32)
    for domain_name, data in domains.items():
        probs = torch.einsum("nec,e->nc", data["probabilities"], w)
        pred = probs.argmax(dim=1)
        result[domain_name] = {
            "accuracy": float((pred == data["labels"]).float().mean().item()),
            "num_samples": int(data["labels"].numel()),
            "mean_weights": weights,
        }
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={device}", flush=True)

    payloads = load_domain_payloads()
    domains = {}
    for name, payload in payloads.items():
        domains[name] = {
            "probabilities": payload["probabilities"].float(),
            "labels": payload["labels"].long(),
        }
        domains[name]["features"] = make_features(domains[name]["probabilities"])
        print(f"loaded {name}: n={domains[name]['labels'].numel()} features={domains[name]['features'].shape[1]}", flush=True)

    static = {
        "testa_fixed": static_metrics(domains, [0.7, 0.2, 0.1]),
        "mnist_template": static_metrics(domains, [0.35, 0.35, 0.30]),
        "balanced_template": static_metrics(domains, [0.50, 0.30, 0.20]),
        "anti1_template": static_metrics(domains, [0.45, 0.45, 0.10]),
    }

    results = []
    for cfg in CONFIGS:
        results.append(run_config(cfg, domains, device))

    best_by_objective = {}
    for objective_name in DOMAIN_WEIGHTS:
        candidates = [result for result in results if result["config"]["objective"] == objective_name]
        if candidates:
            best_by_objective[objective_name] = max(candidates, key=lambda item: item["mean_objective_score"])

    output = {
        "device": device,
        "experts": [{"label": e["label"], "base_weight": e["base_weight"]} for e in EXPERTS],
        "domain_weights": DOMAIN_WEIGHTS,
        "static": static,
        "results": results,
        "best_by_objective": best_by_objective,
    }
    summary_path = OUTPUT_DIR / "learned_router_cv_summary.json"
    summary_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best_by_objective": best_by_objective}, indent=2, ensure_ascii=False))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
