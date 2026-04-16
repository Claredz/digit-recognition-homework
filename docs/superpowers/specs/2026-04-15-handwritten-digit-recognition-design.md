# Handwritten Digit Recognition Project Design

**Date:** 2026-04-15  
**Project type:** Personal course assignment  
**Primary goal:** Build a learning-oriented handwritten digit recognition system that is stable now and easy to optimize later.

## 1. Project Summary

This project is an AI introduction course assignment focused on handwritten digit recognition. Based on the information currently available, the final task will use digit samples collected by the teacher from students' handwriting, and noise may be added to the images. The teacher has also indicated that there will likely be a later stage where the test set can be used for tuning and optimization.

Because the detailed data format, scoring rules, and final submission package are not fully known yet, the project should not optimize prematurely for a specific dataset layout or a one-off high-score solution. Instead, the design should prioritize a clear, reproducible baseline that can be adapted once the final rules and data are released.

A second project goal is explicit learning. The implementation should help the user understand the task, the model, the training loop, and the meaning of evaluation results rather than only producing a finished result quickly.

## 2. Current Assumptions

The current design is based on these assumptions:

1. Source code will definitely need to be submitted.
2. The assignment is individual work.
3. Detailed formal documentation from the teacher is not yet available.
4. The teacher-collected dataset may differ from standard MNIST-style distributions.
5. Noise robustness will matter because noisy samples are expected.
6. A later optimization phase will likely matter, so experiment tracking and tunable structure are important from the start.

If later course instructions contradict any assumption above, the data adapter, experiment configuration, and evaluation/export pieces should be updated before changing the rest of the training pipeline.

## 3. Design Goals

### 3.1 Immediate goals

- Build a full pipeline that can run end to end: data loading, preprocessing, training, validation, evaluation, and prediction.
- Use a small CNN as the main baseline.
- Keep the project structure easy to understand and explain.
- Preserve room for later noise handling and hyperparameter tuning.

### 3.2 Learning goals

- Help the user learn how image classification tasks are structured.
- Make each module understandable on its own.
- Ensure training results can be interpreted, not just produced.
- Support later report writing by preserving experiment evidence.

### 3.3 Long-term goals

- Swap in the teacher's final dataset with minimal structural changes.
- Add noise-oriented preprocessing and augmentation during later optimization.
- Support systematic hyperparameter tuning rather than ad hoc code edits.

## 4. Scope

### 4.1 In scope for the current baseline

- A configurable data input path and dataset adapter.
- Image preprocessing for size normalization, tensor conversion, and normalization.
- Optional basic augmentation hooks for later robustness experiments.
- A small CNN baseline model.
- Train/validation split support.
- Training and validation loops.
- Best-model checkpoint saving.
- Accuracy reporting.
- Confusion matrix and misclassification analysis.
- Batch prediction/export path for future test-set inference.
- Experiment result logging for later tuning.

### 4.2 Out of scope for the current baseline

- Large or complex model families used only to chase peak accuracy immediately.
- Ensembling or multi-model fusion.
- Heavy architecture search.
- Dataset-specific hardcoding tied to the unseen final teacher dataset.
- Premature optimization before a trustworthy baseline exists.

## 5. Recommended Technical Direction

The recommended baseline is a **small CNN-first approach**.

This is preferred over a traditional machine learning only baseline because the task is image recognition, the final dataset may contain noisy handwritten samples, and a CNN provides a stronger default path for later robustness improvements. It also teaches the user the most relevant ideas for this assignment: convolution, feature extraction, training dynamics, overfitting, and evaluation.

Traditional methods such as SVM or KNN can still be added later as optional comparison experiments if needed for report depth, but they are not the main implementation path.

## 6. System Architecture

The system should be divided into focused modules so that each part can be understood, replaced, and tested independently.

### 6.1 Data access layer

Responsibilities:
- Read available image/label data.
- Isolate dataset-specific parsing rules.
- Support later adaptation when the teacher releases the formal dataset.

Reasoning:
Keeping data parsing separate prevents later dataset changes from forcing rewrites across the whole project.

### 6.2 Preprocessing and augmentation layer

Responsibilities:
- Resize or standardize image shape if needed.
- Normalize pixel values.
- Convert images into tensors.
- Provide optional augmentation switches for later experiments.

Likely later additions:
- Light random rotation
- Translation
- Noise injection during training
- Possibly simple denoising preprocessing

Reasoning:
This is the main place where later anti-noise strategies can be introduced without destabilizing the rest of the code.

### 6.3 Model layer

Responsibilities:
- Define a small CNN baseline.
- Keep the architecture simple enough for learning and debugging.
- Make model replacement possible later if stronger variants are needed.

Reasoning:
A small CNN is enough to establish a meaningful baseline while keeping the architecture understandable.

### 6.4 Training layer

Responsibilities:
- Run epoch-based training.
- Track loss and accuracy.
- Evaluate on a validation split.
- Save the best checkpoint.
- Support reproducible training configuration.

Reasoning:
This layer forms the stable experimental core. Once it works, later tuning should mainly change configurations instead of restructuring code.

### 6.5 Evaluation and prediction layer

Responsibilities:
- Compute validation accuracy.
- Generate confusion matrix.
- Save or display misclassified samples.
- Run batch prediction for future formal test sets.
- Export final predictions when needed.

Reasoning:
This layer turns model output into evidence for analysis, optimization, and reporting.

### 6.6 Experiment management layer

Responsibilities:
- Record hyperparameters.
- Record model version and training outcome.
- Save curves, checkpoints, and evaluation artifacts.

Reasoning:
The teacher has already hinted that later tuning matters. Without experiment tracking, that stage will become guesswork.

## 7. Proposed Project Structure

A simple structure is recommended:

```text
project/
├─ data/
├─ src/
│  ├─ data.py
│  ├─ model.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ predict.py
├─ outputs/
│  ├─ checkpoints/
│  ├─ logs/
│  ├─ figures/
│  └─ predictions/
├─ notebooks/
└─ docs/
```

### Structure notes

- `data/` stores raw or prepared datasets.
- `src/data.py` handles loading, transforms, and splitting.
- `src/model.py` defines the CNN.
- `src/train.py` contains the training entry point.
- `src/evaluate.py` handles metrics and error analysis.
- `src/predict.py` supports later test-set prediction export.
- `outputs/` stores artifacts needed for analysis and reporting.
- `notebooks/` can be used only for exploratory analysis, not as the main production path.

This structure supports both learning and later tuning by keeping responsibilities visible.

## 8. Data Strategy

### 8.1 Current phase

Because the final dataset is not fully available, the project should first establish a stable baseline using either:
- any currently available sample data, or
- a standard handwritten digit dataset for pipeline validation.

The goal of this phase is not to finalize the best model. The goal is to validate the full workflow and ensure the system is ready to accept the teacher's dataset later.

### 8.2 Later adaptation phase

When the teacher releases the final dataset or tuning stage:
- adapt only the dataset loading layer if format changes,
- inspect image quality and noise characteristics,
- refine preprocessing and augmentation accordingly,
- rerun controlled experiments using the same training/evaluation pipeline.

This avoids rebuilding the project from scratch.

## 9. Training and Evaluation Design

### 9.1 Training outputs

Each meaningful training run should produce:
- best model checkpoint,
- training and validation loss history,
- training and validation accuracy history,
- run configuration summary,
- optional figure exports of training curves.

### 9.2 Evaluation outputs

Each evaluation should produce:
- validation accuracy,
- confusion matrix,
- representative misclassified samples,
- run notes describing what changed and why.

### 9.3 Why this matters

These outputs serve three purposes at once:
1. confirm the code is behaving reasonably,
2. support future tuning decisions,
3. provide evidence for the final report.

## 10. Learning-Oriented Execution Principle

This project should be implemented in a way that teaches the user while progressing.

That means:
- choosing a simple baseline before stronger variants,
- keeping files focused enough to explain clearly,
- explaining why each major module exists,
- treating evaluation artifacts as learning tools rather than only final metrics,
- using the project structure itself to teach the workflow of an image classification task.

The intended learning progression is:
1. understand the data pipeline,
2. understand the CNN baseline,
3. understand training and validation behavior,
4. understand model errors,
5. understand controlled tuning.

## 11. Risks and Mitigations

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Final dataset format is unknown | A rigid pipeline could break late | Isolate dataset loading and parsing |
| Noise significantly lowers baseline accuracy | Later scoring may depend on robustness | Reserve augmentation and preprocessing hooks now |
| Scoring rules remain unclear | Over-optimizing the wrong thing wastes effort | Build a balanced baseline that is explainable and tunable |
| Working alone increases time pressure | Complex early scope can stall progress | Keep the first model small and stable |
| Tuning becomes chaotic | Late-stage improvements may be hard to compare | Record experiments from the beginning |

## 12. Verification Strategy

The baseline will be considered technically healthy when it can satisfy these checks:

1. Data loads correctly.
2. Labels align with images.
3. Sample visualization looks correct.
4. Training loss decreases meaningfully.
5. Validation accuracy rises above random guessing.
6. Best checkpoint can be saved and reloaded.
7. Evaluation outputs are generated correctly.
8. Prediction can run on a batch input path.

## 13. Phase-Level Success Criteria

### 13.1 Current-stage success

The current stage is successful if:
- the full training pipeline runs end to end,
- the small CNN baseline trains and validates successfully,
- evaluation artifacts are generated,
- the project is organized clearly enough for the user to explain each module,
- later tuning can be added without restructuring everything.

### 13.2 Later-stage success

The later optimization stage is successful if:
- the teacher's final dataset can be integrated quickly,
- noise-oriented experiments can be run systematically,
- hyperparameters can be compared with recorded evidence,
- the final submission is supported by clear analysis rather than guesswork.

## 14. Final Design Statement

The approved design is to build a **learning-oriented, reproducible handwritten digit recognition baseline system centered on a small CNN**, with clear module boundaries, experiment tracking, and explicit support for later adaptation to noisy teacher-provided data and final tuning requirements.

This design intentionally prioritizes clarity, extensibility, and learning value before aggressive optimization.
