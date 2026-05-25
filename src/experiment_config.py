from __future__ import annotations

import json
import subprocess
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
