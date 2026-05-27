# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Hand-written digit recognition for an AI introductory course. Two evaluation domains:

- **MNIST-like** (standard benchmarks): model already achieves ~0.998 accuracy. Checkpoint at `outputs_submission/checkpoints/best_model_state.pt`.
- **TestA** (real-world domain adaptation, the active optimization target): best single-expert OOF ~0.7797, best 5-expert ensemble OOF ~0.7891. Work tracked via `experiments/*.yaml` → `outputs_runs/<experiment_id>/`.

**Critical rule**: never write to or delete `outputs_submission/` — it's the archived high-score submission. All new experiments go to `outputs_runs/<experiment_id>/`.

## Common commands

```bash
# Install dependencies (Python 3.10+)
python -m pip install -r requirements.txt

# Run all tests
python -m pytest

# Quick smoke test training (1 epoch)
python -m src.train --project-root . --dataset-name mnist --model-name medium_cnn --epochs 1 --batch-size 64 --run-name smoke_medium

# Multi-source training
python -m src.train --project-root . --dataset-name multisource --model-name medium_cnn --use-emnist-digits --use-usps --use-qmnist --optimizer-type AdamW --scheduler-type CosineAnnealingLR --run-name final_multisource

# Evaluate a checkpoint on holdout sets
python -m src.evaluate --project-root . --checkpoint <path.pt> --dataset-name mnist --model-name medium_cnn --holdouts mnist_test emnist_digits_test qmnist_test10k

# Predict unlabeled images to CSV
python -m src.predict --project-root . --checkpoint <path.pt> --image-dir <dir> --model-name medium_cnn --use-tta

# TestA: scratch 5-fold baseline
python scripts/run_testa_scratch_kfold.py --config experiments/testa_scratch_5fold.yaml

# TestA: generalist→specialist 5-fold fine-tune
python scripts/run_testa_finetune_kfold.py --config experiments/testa_finetune_from_generalist.yaml

# TestA: specialist out-of-fold evaluation
python scripts/eval_testa_specialist_oof.py --config experiments/testa_specialist_5fold.yaml

# TestA: search specialist+generalist fusion weights
python scripts/predict_testa_fusion.py --config experiments/specialist_generalist_ensemble.yaml --search-only

# All-domain generalist training
python scripts/train_all_domain.py --config experiments/all_domain_medium_generalist_seed42_e80.yaml

# Ensemble OOF evaluation and weight search
python scripts/eval_testa_expert_ensemble_oof.py --two-stage --grid-step 0.05
```

## Architecture

### Core pipeline (`src/`)

```
config.py         → ExperimentConfig dataclass (all hyperparams), ProjectPaths, ensure_project_paths()
data.py           → Dataset/dataloader creation: MNIST/EMNIST/USPS/QMNIST/folder, multi-source merging
model.py          → SmallCNN, MediumCNN, LargeCNN + build_model() factory + normalize_model_name() alias resolver
engine.py         → Training loop (fit, run_epoch), optimizer/scheduler builders, AMP support
train.py          → CLI entry: parses args → ExperimentConfig → create_dataloaders → build_model → fit
evaluate.py       → Checkpoint loading, evaluation on validation/holdout/MNIST-C sets, metric export
predict.py        → Predict unlabeled images with optional TTA, export predictions CSV
preprocess.py     → Image preprocessing (auto-invert, background cleanup, digit cropping, MNIST-style centering)
```

### Model system (`src/model.py` + `src/models/heterogeneous.py`)

`normalize_model_name()` handles alias resolution (e.g., `"medium"` → `"medium_cnn"`, `"vit"` → `"convstem_vit"`). Built-in architectures in `src/model.py`: `SmallCNN`, `MediumCNN`, `LargeCNN` (pure CNN family). Advanced architectures in `src/models/heterogeneous.py`: `PreActResNetTiny` (~1.3M), `WideResNetTiny` (~2.8M, best single model), `ConvNeXtMicro` (~0.8M), `ConvStemViT` (~1.1M), `MobileNetV3_28` (~1.5M). `build_model()` dispatches to heterogeneous models via `HETERO_MODEL_NAMES` lookup.

### Experiment system (`src/experiment_config.py` + `experiments/*.yaml`)

YAML configs with mandatory `experiment_id`. `load_experiment_config()` parses YAML, `resolve_experiment_output_dir()` maps to `outputs_runs/<experiment_id>/`. `assert_safe_output_dir()` guards against writing to `outputs_submission/`. Each run records git commit hash and saves a `config_snapshot.json`.

### TestA specialist pipeline

`src/testa_robust_train.py` is the main TestA training module (~63KB). Supports: 5-fold cross-validation, MixUp/CutMix/RandomErasing, AntiClass1MarginLoss, MNIST-C corruption augmentation, TTA evaluation. Key classes: `TestADataset`, `TestAKFoldSplitter`, training/validation loops with extended diagnostics (per-class metrics, class-1 ratio tracking).

### Losses (`src/losses.py`)

`AntiClass1MarginLoss` adds a margin penalty term to CrossEntropyLoss to suppress class-1 over-prediction (a documented TestA failure mode). Compatible with MixUp — returns a scalar like standard CE so existing training loops compose correctly.

### Data registry (`src/data_registry.py`)

`DomainRegistry` + `DomainSpec` for labeling datasets as domains (e.g., "MNIST_family", "local_digits", "hasyv2"). `DomainBalancedSampler` ensures per-batch domain coverage during multi-domain training.

### Ensemble system (`src/ensemble/`)

`HeuristicDomainRouter` — adjusts per-sample expert weights based on prediction entropy and JS divergence between experts. Used for domain-aware ensemble without learned routing.

### Scripts (`scripts/`)

- `train_all_domain.py` — multi-domain generalist training entry point
- `run_testa_finetune_kfold.py` / `run_testa_scratch_kfold.py` — TestA specialist training
- `eval_testa_specialist_oof.py` — specialist out-of-fold evaluation with diagnostics
- `eval_testa_expert_ensemble_oof.py` — ensemble weight grid search (two-stage)
- `predict_testa_fusion.py` — specialist+generalist weighted fusion prediction
- `domain_aware_ensemble.py` — heuristic domain-aware ensemble prediction
- `calibrate_oof_priors.py` — per-class prior calibration from OOF probabilities

### Output directory conventions

- `outputs_submission/` — archived high-score result (read-only, never overwrite)
- `outputs_runs/<run_name>/` — CLI training runs (`--run-name`)
- `outputs_runs/<experiment_id>/` — YAML-driven experiment runs
- `outputs/` — default fallback output
- Each output dir has: `checkpoints/`, `logs/`, `figures/`, `predictions/`, `evaluation/`

### Notebooks

- `submission_notebook.ipynb` — self-contained submission for teacher (keep runnable standalone)
- `project_notebook.ipynb` — engineering entry point, imports `src/` modules
- `teaching_notebook.ipynb` — pedagogical version with explanations

### Tests (`tests/`)

Run with `python -m pytest`. Covers: config, data loading, model building, training loop, evaluation output, predict preprocessing, ensemble predict, losses, TestA splits. Uses `conftest.py` with a `folder_digit_dataset` fixture.
