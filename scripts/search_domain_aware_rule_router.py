"""Search a first-stage domain-aware rule router.

This script caches averaged expert probabilities per domain once, then searches
rule thresholds with vectorized GPU tensor operations. The router selects among
four weight templates: TestA-hard, MNIST-like, balanced, and anti-class-1.
"""

from __future__ import annotations

import json
import sys
import zipfile
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.data import build_eval_transform
from src.evaluate import load_model_from_checkpoint

EXPERTS = [
    {
        "label": "wide_resnet_tiny_raw_seed42",
        "model_name": "wide_resnet_tiny",
        "base_weight": 0.7,
        "run": "testa_wide_resnet_tiny_raw_seed42_e60",
        "seed": 42,
        "oof": PROJECT_ROOT / "outputs_runs/testa_wide_resnet_tiny_raw_seed42_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_anti1_seed2026",
        "model_name": "medium_cnn",
        "base_weight": 0.2,
        "run": "testa_medium_v2_anti1_margin_seed2026_e60",
        "seed": 2026,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_anti1_margin_seed2026_e60/oof/oof_probabilities.pt",
    },
    {
        "label": "medium_raw_seed3407",
        "model_name": "medium_cnn",
        "base_weight": 0.1,
        "run": "testa_medium_v2_raw_seed3407_e60",
        "seed": 3407,
        "oof": PROJECT_ROOT / "outputs_runs/testa_medium_v2_raw_seed3407_e60/oof/oof_probabilities.pt",
    },
]

TEMPLATES = {
    "testa": [0.70, 0.20, 0.10],
    "mnist": [0.35, 0.35, 0.30],
    "balanced": [0.50, 0.30, 0.20],
    "anti1": [0.45, 0.45, 0.10],
}

OBJECTIVES = {
    "hidden_b_balanced": {"TestA": 0.45, "MNIST-family": 0.35, "MNIST-C": 0.20},
    "hidden_b_easy": {"TestA": 0.35, "MNIST-family": 0.45, "MNIST-C": 0.20},
    "hidden_b_hard": {"TestA": 0.70, "MNIST-family": 0.15, "MNIST-C": 0.15},
}

PARAM_ROWS = [
    {
        "medium_conf": medium_conf,
        "wide_conf": wide_conf,
        "anti1_delta": anti1_delta,
        "wide_slack": wide_slack,
    }
    for medium_conf, wide_conf, anti1_delta, wide_slack in product(
        [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        [0.00, 0.05, 0.10, 0.15, 0.20],
        [0.00, 0.05, 0.10, 0.15],
    )
]

CACHE_DIR = PROJECT_ROOT / "outputs_runs/domain_aware_rule_router/cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs_runs/domain_aware_rule_router"


def checkpoint_paths(expert: dict) -> list[Path]:
    return [
        PROJECT_ROOT / f"outputs_runs/{expert['run']}/seed_{expert['seed']}/fold_{fold}/checkpoints/testa_specialist_best.pt"
        for fold in range(5)
    ]


class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L")
            if self.transform is not None:
                image = self.transform(image)
        return image, target


def load_expert_models(config: ExperimentConfig, device: str):
    loaded = []
    for expert in EXPERTS:
        config.model_name = expert["model_name"]
        models = []
        for checkpoint in checkpoint_paths(expert):
            model, _ = load_model_from_checkpoint(checkpoint, config, device)
            model.eval()
            models.append(model)
        loaded.append({**expert, "models": models})
    return loaded


def averaged_expert_probabilities(experts: list[dict], images: torch.Tensor, device: str) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    expert_probs = []
    with torch.no_grad():
        for expert in experts:
            fold_probs = [torch.softmax(model(images), dim=1) for model in expert["models"]]
            expert_probs.append(torch.stack(fold_probs, dim=0).mean(dim=0))
    return torch.stack(expert_probs, dim=1).cpu()


def cache_dataset_probabilities(name: str, dataset, experts: list[dict], device: str, batch_size: int = 1024) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")
    probabilities = []
    labels = []
    for batch_index, (images, batch_labels) in enumerate(loader, start=1):
        probabilities.append(averaged_expert_probabilities(experts, images, device))
        labels.append(batch_labels.cpu().long())
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            print(f"[cache:{name}] batch {batch_index}/{len(loader)}", flush=True)
    payload = {"probabilities": torch.cat(probabilities, dim=0), "labels": torch.cat(labels, dim=0)}
    torch.save(payload, path)
    return payload


def build_mnist_family_dataset(config: ExperimentConfig):
    return ConcatDataset(
        [
            datasets.MNIST(root=str(config.resolved_data_dir()), train=False, download=True, transform=build_eval_transform(config)),
            datasets.EMNIST(
                root=str(config.resolved_data_dir()),
                split="digits",
                train=False,
                download=True,
                transform=build_eval_transform(config, correct_emnist=True),
            ),
            datasets.QMNIST(
                root=str(config.resolved_data_dir()),
                what="test10k",
                compat=True,
                download=True,
                transform=build_eval_transform(config),
            ),
            datasets.USPS(root=str(config.resolved_data_dir()), train=True, download=False, transform=build_eval_transform(config)),
        ]
    )


def build_external_dataset(config: ExperimentConfig):
    transform = build_eval_transform(config)
    return ConcatDataset(
        [
            GrayscaleImageFolder(str(PROJECT_ROOT / "data/penbased_rendered"), transform=transform),
            GrayscaleImageFolder(str(PROJECT_ROOT / "data/hasyv2_digits"), transform=transform),
            GrayscaleImageFolder(str(PROJECT_ROOT / "data/chars74k_digits"), transform=transform),
        ]
    )


def build_mnist_c_tensor_dataset(config: ExperimentConfig):
    zip_path = config.resolved_data_dir() / "mnist_c" / "mnist_c.zip"
    image_chunks = []
    label_chunks = []
    names = []
    with zipfile.ZipFile(zip_path) as archive:
        corruptions = sorted(
            {
                parts[1]
                for item in archive.namelist()
                if (parts := item.split("/")) and len(parts) == 3 and parts[2] == "test_images.npy"
            }
        )
        for corruption in corruptions:
            with archive.open(f"mnist_c/{corruption}/test_images.npy") as image_file:
                images_np = np.load(image_file)
            with archive.open(f"mnist_c/{corruption}/test_labels.npy") as label_file:
                labels_np = np.load(label_file)
            if images_np.ndim == 4 and images_np.shape[-1] == 1:
                images_np = np.squeeze(images_np, axis=-1)
            images = torch.from_numpy(images_np).float().unsqueeze(1) / 255.0
            images = (images - 0.5) / 0.5
            labels = torch.from_numpy(labels_np).long()
            image_chunks.append(images)
            label_chunks.append(labels)
            names.extend([corruption] * len(labels))
    return TensorDataset(torch.cat(image_chunks, dim=0), torch.cat(label_chunks, dim=0)), names


def load_testa_oof() -> dict:
    combined_labels = None
    combined_sample_ids = None
    probabilities = []
    for expert in EXPERTS:
        payload = torch.load(expert["oof"], map_location="cpu")
        labels = payload["labels"].long()
        if combined_labels is None:
            combined_labels = labels
            combined_sample_ids = payload["sample_ids"]
        elif combined_sample_ids != payload["sample_ids"] or not torch.equal(combined_labels, labels):
            raise ValueError(f"OOF order mismatch for {expert['label']}")
        probabilities.append(payload["probabilities"].float())
    return {"probabilities": torch.stack(probabilities, dim=1), "labels": combined_labels}


def static_predictions(probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nec,e->nc", probs, weights).argmax(dim=1)


def route_counts_for_params(
    probs: torch.Tensor,
    labels: torch.Tensor,
    params: torch.Tensor,
    templates: torch.Tensor,
    device: str,
    param_batch_size: int = 64,
    sample_batch_size: int = 8192,
) -> torch.Tensor:
    correct_counts = torch.zeros(params.shape[0], device=device, dtype=torch.float32)
    labels_device = labels.to(device)
    probs_device = probs.to(device)

    for param_start in range(0, params.shape[0], param_batch_size):
        param_chunk = params[param_start : param_start + param_batch_size]
        chunk_correct = torch.zeros(param_chunk.shape[0], device=device, dtype=torch.float32)
        medium_conf_th = param_chunk[:, 0].view(-1, 1)
        wide_conf_th = param_chunk[:, 1].view(-1, 1)
        anti1_delta_th = param_chunk[:, 2].view(-1, 1)
        wide_slack = param_chunk[:, 3].view(-1, 1)

        for sample_start in range(0, probs_device.shape[0], sample_batch_size):
            p = probs_device[sample_start : sample_start + sample_batch_size]
            y = labels_device[sample_start : sample_start + sample_batch_size]

            max_probs, expert_preds = p.max(dim=-1)
            wide_conf = max_probs[:, 0].view(1, -1)
            medium_conf = ((max_probs[:, 1] + max_probs[:, 2]) * 0.5).view(1, -1)
            medium_agree = (expert_preds[:, 1] == expert_preds[:, 2]).view(1, -1)
            wide_agrees_medium = ((expert_preds[:, 0] == expert_preds[:, 1]) | (expert_preds[:, 0] == expert_preds[:, 2])).view(1, -1)

            fixed_probs = torch.einsum("nec,e->nc", p, templates[0])
            fixed_pred = fixed_probs.argmax(dim=1).view(1, -1)
            anti1_pred = expert_preds[:, 1].view(1, -1)
            anti1_delta = (fixed_probs[:, 1] - p[:, 1, 1]).view(1, -1)

            mnist_mask = medium_agree & (medium_conf >= medium_conf_th) & (wide_conf <= medium_conf + wide_slack)
            testa_mask = (wide_conf >= wide_conf_th) & wide_agrees_medium
            anti1_mask = (fixed_pred == 1) & (anti1_pred != 1) & (anti1_delta >= anti1_delta_th)

            template_index = torch.full_like(mnist_mask, fill_value=2, dtype=torch.long)
            template_index = torch.where(testa_mask, torch.full_like(template_index, 0), template_index)
            template_index = torch.where(mnist_mask, torch.full_like(template_index, 1), template_index)
            template_index = torch.where(anti1_mask, torch.full_like(template_index, 3), template_index)
            weights = templates[template_index]

            ensemble_probs = (weights.unsqueeze(-1) * p.unsqueeze(0)).sum(dim=2)
            predictions = ensemble_probs.argmax(dim=-1)
            chunk_correct += (predictions == y.view(1, -1)).float().sum(dim=1)
        correct_counts[param_start : param_start + param_chunk.shape[0]] = chunk_correct
    return correct_counts


def evaluate_static(probs: torch.Tensor, labels: torch.Tensor, template_weights: list[float]) -> float:
    weights = torch.tensor(template_weights, dtype=torch.float32)
    pred = static_predictions(probs, weights)
    return float((pred == labels).float().mean().item())


def evaluate_selected_rule(probs: torch.Tensor, labels: torch.Tensor, param_row: dict, templates: torch.Tensor, device: str) -> dict:
    params = torch.tensor(
        [[param_row["medium_conf"], param_row["wide_conf"], param_row["anti1_delta"], param_row["wide_slack"]]],
        device=device,
        dtype=torch.float32,
    )
    correct = route_counts_for_params(probs, labels, params, templates, device)[0]
    return {"accuracy": float((correct / len(labels)).item()), "num_samples": int(len(labels))}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(project_root=PROJECT_ROOT, verbose=False, external_validation_batch_size=1024)
    templates = torch.tensor([TEMPLATES[name] for name in ["testa", "mnist", "balanced", "anti1"]], device=device, dtype=torch.float32)
    params = torch.tensor(
        [[row["medium_conf"], row["wide_conf"], row["anti1_delta"], row["wide_slack"]] for row in PARAM_ROWS],
        device=device,
        dtype=torch.float32,
    )

    print(f"device={device} params={len(PARAM_ROWS)}", flush=True)
    experts = load_expert_models(config, device)

    domain_payloads = {
        "TestA": load_testa_oof(),
        "MNIST-family": cache_dataset_probabilities("mnist_family", build_mnist_family_dataset(config), experts, device),
        "MNIST-C": cache_dataset_probabilities("mnist_c", build_mnist_c_tensor_dataset(config)[0], experts, device),
        "local/external digits": cache_dataset_probabilities("external_digits", build_external_dataset(config), experts, device),
    }

    static_results = {}
    for domain_name, payload in domain_payloads.items():
        static_results[domain_name] = {
            template_name: evaluate_static(payload["probabilities"], payload["labels"], weights)
            for template_name, weights in TEMPLATES.items()
        }

    search_domains = ["TestA", "MNIST-family", "MNIST-C"]
    rule_accuracies = {}
    for domain_name in search_domains:
        print(f"[search] {domain_name}", flush=True)
        payload = domain_payloads[domain_name]
        counts = route_counts_for_params(payload["probabilities"], payload["labels"], params, templates, device)
        rule_accuracies[domain_name] = counts / len(payload["labels"])

    objective_results = {}
    for objective_name, weights in OBJECTIVES.items():
        score = sum(rule_accuracies[domain] * domain_weight for domain, domain_weight in weights.items())
        best_value, best_index_tensor = torch.max(score, dim=0)
        best_index = int(best_index_tensor.item())
        best_params = PARAM_ROWS[best_index]
        domain_eval = {
            domain_name: evaluate_selected_rule(payload["probabilities"], payload["labels"], best_params, templates, device)
            for domain_name, payload in domain_payloads.items()
        }
        objective_results[objective_name] = {
            "objective_score": float(best_value.item()),
            "params": best_params,
            "domain_eval": domain_eval,
        }

    output = {
        "experts": [{"label": expert["label"], "base_weight": expert["base_weight"]} for expert in EXPERTS],
        "templates": TEMPLATES,
        "objectives": OBJECTIVES,
        "static_results": static_results,
        "objective_results": objective_results,
    }
    out_path = OUTPUT_DIR / "rule_router_summary.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
