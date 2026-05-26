from __future__ import annotations

import datetime
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import yaml


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"实验配置必须是 YAML mapping: {config_path}")
    if not payload.get("experiment_id"):
        raise ValueError(f"实验配置缺少 experiment_id: {config_path}")
    payload["_config_path"] = str(config_path)
    return payload


def resolve_project_root(project_root: Path | str | None = None) -> Path:
    if project_root is None:
        return Path(__file__).resolve().parents[1]
    return Path(project_root).expanduser().resolve()


def resolve_path(value: str | Path | None, project_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def resolve_experiment_output_dir(config: dict[str, Any], project_root: Path) -> Path:
    if config.get("output_dir"):
        return resolve_path(config["output_dir"], project_root)  # type: ignore[return-value]
    output_root = Path(str(config.get("output_root", "outputs_runs")))
    if output_root.is_absolute():
        root = output_root
    else:
        root = project_root / output_root
    return (root / str(config["experiment_id"])).resolve()


def assert_safe_output_dir(output_dir: Path, project_root: Path, allow_outputs_submission: bool = False) -> None:
    output_dir = output_dir.resolve()
    archive_dir = (project_root / "outputs_submission").resolve()
    if allow_outputs_submission:
        return
    if output_dir == archive_dir or archive_dir in output_dir.parents:
        raise ValueError(
            "新 TestA 实验默认不能写入 outputs_submission/；"
            "请改用 outputs_runs/<experiment_id>/，或显式启用 allow_outputs_submission。"
        )


def git_commit_hash(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def write_config_snapshot(config: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {key: value for key, value in config.items() if not key.startswith("_")}
    path = output_dir / "config_snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"配置段 {name} 必须是 mapping")
    return section


def config_list(config: dict[str, Any], name: str, default: list[Any]) -> list[Any]:
    value = config.get(name, default)
    if isinstance(value, list):
        return value
    return [value]


# ---------------------------------------------------------------------------
# Schema validator (Phase A.1)
#
# Goal:
#   1. Catch typos like `mixup_alfa` instead of `mixup_alpha` before training starts.
#   2. Recognise newer, opt-in sections (anti_class1_loss / data.domains / ensemble.*)
#      that arrive with the multi-expert system.
#   3. Stay BACKWARD COMPATIBLE: every existing YAML in experiments/ must validate
#      with zero warnings. Unknown keys produce warnings, not errors, unless strict=True.
# ---------------------------------------------------------------------------

KNOWN_TOP_LEVEL_KEYS: set[str] = {
    "experiment_id",
    "output_root",
    "output_dir",
    "seed",
    "seeds",
    "folds",
    "model",
    "training",
    "augmentation",
    "ensemble",
    "prediction",
    "data",
    "_config_path",  # internal marker injected by load_experiment_config
}

KNOWN_MODEL_KEYS: set[str] = {
    "model_name",
    "dropout",
    "init_checkpoint",
    "init_checkpoint_template",
    "generalist_checkpoint",
    "aliases",  # new (Phase A.1): allow alias registration metadata
}

KNOWN_TRAINING_KEYS: set[str] = {
    "mode",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "label_smoothing",
    "patience",
    "freeze_backbone_epochs",
    "use_amp",
    "allow_tf32",
    "num_workers",
    "checkpoint_name",
    "anti_class1_loss",  # new (Phase B.2)
}

KNOWN_ANTI_CLASS1_KEYS: set[str] = {"enabled", "lambda", "margin", "target_class"}

KNOWN_AUGMENTATION_KEYS: set[str] = {
    "preprocessing_mode",
    "evaluate_preprocess",
    "mixup_alpha",
    "cutmix_alpha",
    "mix_prob",
    "random_erasing_p",
    "affine_degrees",
    "translate_ratio",
    "scale_min",
    "scale_max",
    "shear_degrees",
    "use_testa_like_augment",
    "testa_like_augment_strength",
    "testa_like_morph_p",
    "testa_like_dilate_bias",
    "testa_like_blur_p",
    "testa_like_contrast_p",
    "preprocess_probability",  # observed in build_fold_config
}

KNOWN_ENSEMBLE_KEYS: set[str] = {
    "fusion_weights",
    "specialist_checkpoint_name",
    "experiments",  # new (Phase A.3): list of experiment_ids to fuse
    "grid_step",
    "weight_search_strategy",  # "uniform" / "grid" / "two_stage" / "dirichlet"
    "max_combinations",
    "n_dirichlet_samples",
}

KNOWN_PREDICTION_KEYS: set[str] = {
    "preprocessing_mode",
    "tta_n",
    "batch_size",
}

KNOWN_DATA_KEYS: set[str] = {
    "domains",  # new (Phase C.1): list of {name, path, weight, loader_kind}
    "sampling",  # new (Phase C.1): {mode, samples_per_domain_per_batch}
    "target_domain",  # new (Phase C.4): for domain-specialist configs
    "num_folds",
}

KNOWN_DOMAIN_KEYS: set[str] = {"name", "path", "weight", "loader_kind"}
KNOWN_SAMPLING_KEYS: set[str] = {"mode", "samples_per_domain_per_batch"}

VALID_TRAINING_MODES: set[str] = {"testa_finetune", "testa_scratch", "testa_specialist"}
VALID_SAMPLING_MODES: set[str] = {"random", "domain_balanced", "domain_weighted"}


def _check_known_keys(section_name: str, payload: dict[str, Any], known: set[str], warnings_out: list[str]) -> None:
    extra = set(payload.keys()) - known
    if extra:
        warnings_out.append(
            f"[{section_name}] 未知字段: {sorted(extra)} (拼写错误?); "
            f"已知合法字段: {sorted(known)}"
        )


def _check_anti_class1(section: dict[str, Any], warnings_out: list[str]) -> None:
    if not isinstance(section, dict):
        warnings_out.append("[training.anti_class1_loss] 必须是 mapping")
        return
    _check_known_keys("training.anti_class1_loss", section, KNOWN_ANTI_CLASS1_KEYS, warnings_out)
    if section.get("enabled"):
        lam = section.get("lambda")
        margin = section.get("margin")
        if not isinstance(lam, (int, float)) or lam < 0:
            warnings_out.append("[training.anti_class1_loss.lambda] 应为非负数")
        if not isinstance(margin, (int, float)):
            warnings_out.append("[training.anti_class1_loss.margin] 应为数值")
        tgt = section.get("target_class", 1)
        if not isinstance(tgt, int) or not 0 <= tgt <= 9:
            warnings_out.append("[training.anti_class1_loss.target_class] 应为 0..9 之间的整数")


def _check_domains(domains: Any, warnings_out: list[str]) -> None:
    if domains is None:
        return
    if not isinstance(domains, list):
        warnings_out.append("[data.domains] 必须是列表")
        return
    for index, dom in enumerate(domains):
        if not isinstance(dom, dict):
            warnings_out.append(f"[data.domains[{index}]] 必须是 mapping")
            continue
        _check_known_keys(f"data.domains[{index}]", dom, KNOWN_DOMAIN_KEYS, warnings_out)
        if "name" not in dom:
            warnings_out.append(f"[data.domains[{index}]] 缺少 name 字段")
        if "path" not in dom:
            warnings_out.append(f"[data.domains[{index}]] 缺少 path 字段")
        weight = dom.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or weight < 0:
            warnings_out.append(f"[data.domains[{index}].weight] 应为非负数")


def _check_sampling(sampling: Any, warnings_out: list[str]) -> None:
    if sampling is None:
        return
    if not isinstance(sampling, dict):
        warnings_out.append("[data.sampling] 必须是 mapping")
        return
    _check_known_keys("data.sampling", sampling, KNOWN_SAMPLING_KEYS, warnings_out)
    mode = sampling.get("mode", "random")
    if mode not in VALID_SAMPLING_MODES:
        warnings_out.append(f"[data.sampling.mode] 必须是 {sorted(VALID_SAMPLING_MODES)} 之一; 实际 = {mode!r}")


def validate_experiment_config(
    payload: dict[str, Any],
    *,
    strict: bool = False,
    emit: bool = True,
) -> list[str]:
    """Schema-validate an experiment config payload.

    Returns the list of warnings collected. When ``strict=True``, raises
    ``ValueError`` if any warning is produced; otherwise just emits warnings via
    ``warnings.warn`` (and the caller can read the returned list).

    Designed to be 100% backward compatible: every YAML currently in
    ``experiments/`` must produce zero warnings. Future new sections (anti_class1_loss,
    data.domains, ensemble.experiments, ...) are recognised here as opt-in extensions.
    """
    if not isinstance(payload, dict):
        raise ValueError("配置必须是 mapping")
    if not payload.get("experiment_id"):
        raise ValueError("配置缺少 experiment_id")

    warnings_out: list[str] = []

    _check_known_keys("top-level", payload, KNOWN_TOP_LEVEL_KEYS, warnings_out)

    folds = payload.get("folds")
    if folds is not None and (not isinstance(folds, int) or folds < 1):
        warnings_out.append(f"[folds] 应为正整数; 实际 = {folds!r}")

    seeds = payload.get("seeds", [])
    if seeds and not isinstance(seeds, list):
        warnings_out.append("[seeds] 应为列表")

    model = payload.get("model", {})
    if isinstance(model, dict):
        _check_known_keys("model", model, KNOWN_MODEL_KEYS, warnings_out)

    training = payload.get("training", {})
    if isinstance(training, dict):
        _check_known_keys("training", training, KNOWN_TRAINING_KEYS, warnings_out)
        mode = training.get("mode")
        if mode is not None and mode not in VALID_TRAINING_MODES:
            warnings_out.append(
                f"[training.mode] 应为 {sorted(VALID_TRAINING_MODES)} 之一; 实际 = {mode!r}"
            )
        if "anti_class1_loss" in training:
            _check_anti_class1(training["anti_class1_loss"], warnings_out)

    augmentation = payload.get("augmentation", {})
    if isinstance(augmentation, dict):
        _check_known_keys("augmentation", augmentation, KNOWN_AUGMENTATION_KEYS, warnings_out)

    ensemble = payload.get("ensemble", {})
    if isinstance(ensemble, dict):
        _check_known_keys("ensemble", ensemble, KNOWN_ENSEMBLE_KEYS, warnings_out)

    prediction = payload.get("prediction", {})
    if isinstance(prediction, dict):
        _check_known_keys("prediction", prediction, KNOWN_PREDICTION_KEYS, warnings_out)

    data = payload.get("data", {})
    if isinstance(data, dict):
        _check_known_keys("data", data, KNOWN_DATA_KEYS, warnings_out)
        _check_domains(data.get("domains"), warnings_out)
        _check_sampling(data.get("sampling"), warnings_out)

    if strict and warnings_out:
        raise ValueError(
            "实验配置 schema 校验失败 (strict=True):\n  - " + "\n  - ".join(warnings_out)
        )

    if emit:
        for message in warnings_out:
            warnings.warn(f"[config] {message}", stacklevel=2)

    return warnings_out


# ---------------------------------------------------------------------------
# Unified manifest writer (Phase A.1)
#
# Writes outputs_runs/<exp>/manifest.json next to (NOT replacing) the existing
# aggregate_summary.json. Per the plan, this is the canonical lookup for
# downstream tooling (extended diagnostics, ensemble scripts, domain-aware
# router) to discover what models / checkpoints / seeds are inside a run.
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA_VERSION = "1.0"


def write_run_manifest(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    """Write outputs_runs/<exp>/manifest.json with a schema-versioned record.

    ``payload`` is merged into a base manifest containing:
      - schema_version
      - created_at (UTC iso8601)
      - git_commit (best-effort)

    Existing manifest.json is overwritten — the file is a derived artifact.
    aggregate_summary.json is NEVER touched by this function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = resolve_project_root(payload.get("project_root"))
    base: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "git_commit": git_commit_hash(project_root),
    }
    base.update({k: v for k, v in payload.items() if k != "project_root"})
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(base, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_run_manifest(manifest_path: Path) -> dict[str, Any]:
    """Best-effort manifest loader.

    Tries manifest.json first (new schema); falls back to aggregate_summary.json
    so that existing experiment outputs (which only have aggregate_summary.json)
    can still be discovered by downstream tooling.
    """
    manifest_path = Path(manifest_path)
    if manifest_path.is_dir():
        candidate = manifest_path / "manifest.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
        fallback = manifest_path / "aggregate_summary.json"
        if fallback.exists():
            payload = json.loads(fallback.read_text(encoding="utf-8"))
            payload.setdefault("schema_version", "legacy_aggregate")
            return payload
        raise FileNotFoundError(f"未找到 manifest.json 或 aggregate_summary.json: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))
