from __future__ import annotations

from pathlib import Path

from run_testa_finetune_kfold import PROJECT_ROOT, run_from_config


if __name__ == "__main__":
    run_from_config(
        default_config=PROJECT_ROOT / "experiments" / "testa_scratch_5fold.yaml",
        forced_mode="testa_scratch",
    )
