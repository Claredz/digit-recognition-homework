# 手写数字识别作业（Notebook 提交版 + 工程复现版）

这是一个面向 AI 导论课程的手写数字识别项目。项目同时保留：

1. **完整独立提交 notebook**：`submission_notebook.ipynb`，适合单独交给老师。
2. **工程化复现入口**：`project_notebook.ipynb` 和 `src/`，支持训练、评估、预测分开运行。
3. **高分结果归档**：`outputs_submission/`，保存上一版约 99.8% 准确率的模型参数、配置、日志、图表和预测结果。

## 重要说明：不要覆盖 `outputs_submission/`

`outputs_submission/` 是上一版高分提交结果归档，包含：

- `outputs_submission/checkpoints/best_model_state.pt`
- `outputs_submission/logs/history.json`
- `outputs_submission/logs/evaluation_summary.json`
- `outputs_submission/hpo/best_params.json`
- `outputs_submission/figures/confusion_matrix.png`
- `outputs_submission/predictions/*.csv`

其中 `evaluation_summary.json` 记录的 mixed validation accuracy 约为 **0.9987**。新实验默认应写入 `outputs/` 或 `outputs_runs/<run_name>/`，不要写入或清空 `outputs_submission/`。

## 当前优化主线：TestA 目标域适配

请把项目里的两个指标分开理解：

1. **MNIST-like benchmark**：`outputs_submission/logs/evaluation_summary.json` 里的约 **0.998** accuracy，说明模型已经具备很强的普通手写数字识别能力。
2. **TestA benchmark**：旧 TestA mixed robust / K-Fold 路线约 **0.72**；最新 TestA-partial 初始化 specialist 路线的可信 OOF accuracy 为 **0.7420**，full TestA raw+TTA 5-fold ensemble 为 **0.7831**。

这几个数来自不同评估方式，不能直接当作同一个测试集上的高低对比：MNIST-like 的 0.998 是普通手写数字分布；TestA OOF 的 0.7420 是每个样本由没见过它的 fold 模型预测，更适合作为可信验证；full TestA 的 0.7831 会更乐观，更适合看最终冲分潜力。

### 当前最好 TestA 结果

目前最强的 TestA 路线是：

```text
robust_expert_v2_testa_partial_best_epoch12_score07098.pt
        ↓
5-Fold TestA-only specialist fine-tuning
        ↓
40 epoch, lr=1e-4, MixUp=0.1, RandomErasing=0.05, CutMix=0
        ↓
raw view + TTA
        ↓
specialist-only probability mean
```

关键结果：

| 评估方式 | 结果 | 说明 |
|---|---:|---|
| TestA-partial init + 40 epoch specialist OOF | **0.7420** | 最可信的 TestA 内部验证指标 |
| TestA-partial init + 5-fold raw-only full TestA ensemble | **0.7777** | full TestA 上的集成结果，偏乐观 |
| TestA-partial init + 5-fold raw+TTA full TestA ensemble | **0.7831** | 当前 full TestA 最好结果，偏乐观 |
| pre-TestA robust init + 40 epoch specialist OOF | 0.7026 | 更严格初始化，但分数较低 |
| 旧 mixed robust K-Fold raw+TTA | 约 0.7223 | 旧路线结果 |

融合搜索显示 specialist + generalist 加权时最优权重为 `w=1.0`，即 **specialist-only** 最好；generalist 概率融合没有提升 OOF。

推荐路线：

```text
generalist pretraining / robust training
        ↓
严格 TestA-only specialist fine-tuning
        ↓
5-Fold TestA specialist ensemble
        ↓
specialist ensemble + generalist weighted fusion
```

新 TestA specialist 路线默认遵循：

- 使用未见过 TestA 标签的 generalist checkpoint 初始化，保证 OOF 验证更可信。
- 每个 fold 只使用 TestA train fold 训练，只用 TestA val fold 验证。
- raw image 是默认主视图；preprocess 只作为诊断，不参与默认选最佳 checkpoint。
- 默认关闭 CutMix 和 RandomErasing，使用保守增强。
- 新实验默认写入 `outputs_runs/<experiment_id>/`，不覆盖 `outputs_submission/`。

### TestA specialist 命令示例

TestA-only scratch 5-Fold baseline：

```bash
python scripts/run_testa_scratch_kfold.py --config experiments/testa_scratch_5fold.yaml
```

Generalist -> TestA specialist 5-Fold fine-tune：

```bash
python scripts/run_testa_finetune_kfold.py --config experiments/testa_finetune_from_generalist.yaml
```

生成 specialist out-of-fold 评估与错误分析：

```bash
python scripts/eval_testa_specialist_oof.py --config experiments/testa_specialist_5fold.yaml
```

搜索 specialist + generalist 融合权重：

```bash
python scripts/predict_testa_fusion.py --config experiments/specialist_generalist_ensemble.yaml --search-only
```

最终融合预测示例：

```bash
python scripts/predict_testa_fusion.py --config experiments/specialist_generalist_ensemble.yaml --image-dir <无标签图片目录>
```

## 推荐使用方式

### 1. 提交给老师

使用：

```text
submission_notebook.ipynb
```

这个 notebook 保持自包含，适合老师只看一个 notebook 的场景。它应保留完整的数据处理、模型定义、训练、评估、预测、图表和结论。

### 2. 项目内分步复现

使用：

```text
project_notebook.ipynb
```

这个 notebook 是工程主入口，主要调用 `src/` 模块。它不是提交版本，适合你自己调试、复现实验、单独跑训练/评估/预测。

### 3. 命令行运行

工程代码也支持命令行分步运行：

- 只训练：`python -m src.train ...`
- 只评估：`python -m src.evaluate ...`
- 只预测：`python -m src.predict ...`

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── pytest.ini
├── submission_notebook.ipynb        # 完整独立提交版 notebook
├── project_notebook.ipynb           # 工程化分步复现 notebook
├── teaching_notebook.ipynb          # 教学解释版 notebook
├── src/
│   ├── config.py                    # 实验配置与输出目录管理
│   ├── data.py                      # MNIST / EMNIST / USPS / QMNIST / folder 数据加载
│   ├── model.py                     # SmallCNN、MediumCNN、build_model
│   ├── engine.py                    # 训练循环、优化器、调度器、early stopping
│   ├── train.py                     # 命令行训练入口
│   ├── evaluate.py                  # 命令行评估入口
│   └── predict.py                   # 无标签图片预测与 CSV 导出
├── tests/                           # 自动化测试
├── outputs/                         # 默认实验输出
├── outputs_runs/                    # run_name 实验输出
└── outputs_submission/              # 99.8% 高分结果归档，不要覆盖
```

## 环境准备

建议使用 Python 3.10 或更高版本。

```bash
python -m pip install -r requirements.txt
```

## 模型说明

项目保留两类模型：

- `small_cnn`：轻量基线，适合教学和快速 smoke test。
- `medium_cnn`：提交版主模型，结构与 `outputs_submission/checkpoints/best_model_state.pt` 的高分 checkpoint 兼容。

`medium_cnn` 使用多层卷积、BatchNorm、AdaptiveAvgPool 和 Dropout，适合多源手写数字训练。

## 数据源

工程代码支持：

- MNIST
- EMNIST Digits
- USPS
- QMNIST
- folder 格式的自定义带标签数据集
- 无标签考试图片目录

EMNIST Digits 会单独应用方向修正，不会影响其他数据源。

## 分步运行示例

### 快速 baseline 训练

```bash
python -m src.train --project-root . --dataset-name mnist --model-name small_cnn --epochs 1 --batch-size 64 --run-name smoke_small
```

输出会写入：

```text
outputs_runs/smoke_small/
```

### 快速 medium_cnn 训练 smoke test

```bash
python -m src.train --project-root . --dataset-name mnist --model-name medium_cnn --epochs 1 --batch-size 64 --run-name smoke_medium
```

### 多源训练入口

```bash
python -m src.train --project-root . --dataset-name multisource --model-name medium_cnn --use-emnist-digits --use-usps --use-qmnist --optimizer-type AdamW --scheduler-type CosineAnnealingLR --run-name final_multisource
```

第一次运行会下载 torchvision 数据集，耗时较长。Windows 下如果 DataLoader 卡住，可把 `--num-workers` 设为 `0` 或较小值。

### 评估模型

评估普通训练输出：

```bash
python -m src.evaluate --project-root . --checkpoint outputs_runs/smoke_medium/checkpoints/best_model.pt --dataset-name mnist --model-name medium_cnn
```

评估上一版高分 checkpoint 的快速 validation smoke split：

```bash
python -m src.evaluate --project-root . --checkpoint outputs_submission/checkpoints/best_model_state.pt --dataset-name mnist --model-name medium_cnn --output-dir outputs_runs/eval_high_score
```

注意：如果配置里用了 `max_samples` 或 notebook 默认小样本设置，这里的 validation accuracy 可能显示 `1.0`，它只说明小样本流程跑通，不代表正式测试集结果。

评估多个正式 holdout 测试集：

```bash
python -m src.evaluate --project-root . --checkpoint outputs_submission/checkpoints/best_model_state.pt --dataset-name mnist --model-name medium_cnn --output-dir outputs_runs/eval_high_score --holdouts mnist_test emnist_digits_test qmnist_test10k
```

如果要额外评估 MNIST-C corruption zip：

```bash
python -m src.evaluate --project-root . --checkpoint outputs_submission/checkpoints/best_model_state.pt --dataset-name mnist --model-name medium_cnn --output-dir outputs_runs/eval_high_score --include-mnist-c
```

评估结果会写入：

```text
outputs_runs/eval_high_score/evaluation/
```

包括：

- `validation/summary.json`
- `holdouts/external_holdouts_summary.json`
- `holdouts/external_holdouts_summary.csv`
- `mnist_c/mnist_c_corruption_summary.csv`
- 各数据集自己的 `classification_report.json`、`confusion_matrix.csv`、`confusion_matrix.png`

### 对无标签图片预测

```bash
python -m src.predict --project-root . --checkpoint outputs_submission/checkpoints/best_model_state.pt --image-dir <无标签图片目录> --model-name medium_cnn --output-dir outputs_runs/predict_high_score --use-tta
```

输出：

```text
outputs_runs/predict_high_score/predictions/predictions.csv
```

CSV 格式：

```csv
filename,prediction
sample_001.png,7
sample_002.png,3
```

## Notebook 分段运行建议

`submission_notebook.ipynb` 应保留可单独提交的完整内容。为了避免每次都重跑耗时步骤，建议在 notebook 中使用开关：

```python
RUN_TRAINING = False
RUN_HPO = False
RUN_EXTERNAL_EVAL = True
RUN_HOLDOUTS = True
RUN_MNIST_C = False
RUN_PREDICTION = True
LOAD_EXISTING_BEST_MODEL = True
```

这样可以跳过训练，直接加载 `outputs_submission/checkpoints/best_model_state.pt` 做评估或预测。`project_notebook.ipynb` 里 `validation_smoke` 是小样本流程检查；正式指标应看 `holdouts` 和可选的 `mnist_c` 汇总。

`project_notebook.ipynb` 默认使用工程模块和 `outputs_runs/project_notebook/`，适合调试，不会覆盖高分结果。

## 自动化测试

运行：

```bash
python -m pytest
```

测试覆盖配置、数据加载、模型、训练循环、评估输出和预测预处理。

## 提交建议

课程作业通常建议提交：

```text
submission_notebook.ipynb
README.md
必要截图或实验报告
预测 CSV（如果老师要求）
```

## 多专家异构系统（Multi-Expert System）

详见 [`docs/final_expert_system_plan.md`](docs/final_expert_system_plan.md)。

### 训练最终结果（全部完成：2026-05-27）

| 排名 | 实验 | OOF acc | class1 ratio | class8 acc | X→1 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 🥇 | **5-expert final ensemble** | **0.7766** | 1.2228 | **0.7251** | 131 | 最终提交 |
| 🥈 | `preact_resnet_tiny_anti1_seed42` | **0.7569** | **1.1503** | 0.6949 | **110** | 最强单专家 |
| 🥉 | `preact_resnet_tiny_raw_seed42` | 0.7514 | 1.2021 | 0.6918 | 127 | 异构突破 |
| 4 | `medium_anti1_seed42` | 0.7474 | 1.2098 | 0.6798 | 146 | |
| 5 | `medium_anti1_seed3407` | 0.7477 | 1.2435 | 0.6858 | 161 | |
| 6 | `medium_raw_seed3407` | 0.7471 | 1.2720 | 0.6918 | 168 | |
| 7 | `medium_anti1_seed2026` | 0.7451 | 1.1969 | 0.6888 | 144 | bias 最佳 |
| 8 | 基线 `testa_partial_e40` | 0.7420 | 1.2746 | 0.6918 | 169 | 参照 |
| ❌ | `large_cnn` | 0.6823 | 1.2927 | 0.6375 | 178 | 淘汰 |
| ❌ | `convnext_micro` | 0.6129 | 1.1036 | 0.4804 | 149 | 淘汰 |

**相对基线**: OOF +0.0346, class1 ratio 1.275→1.223, class8 0.692→0.725, X→1 169→131。

**核心策略**: 异构架构(PreActResNet) + anti-class-1 margin loss + multi-seed ensemble。

### 新增架构

| 架构 | 参数量 | 说明 |
|---|---|---|
| `preact_resnet_tiny` | ~1.3M | PreAct ResNet 风格 |
| `wide_resnet_tiny` | ~2.8M | wider channels |
| `convnext_micro` | ~0.8M | depthwise + LayerNorm |
| `convstem_vit` | ~1.1M | CNN stem + Transformer |
| `mobilenetv3_28` | ~1.5M | inverted residual + SE |

### 新功能

- `scripts/eval_testa_specialist_oof.py --baseline-experiment-id <id>` 输出扩展诊断 JSON
- `scripts/eval_testa_expert_ensemble_oof.py --two-stage --grid-step 0.05` 两段式网格搜索
- `scripts/domain_aware_ensemble.py` 启发式域感知集成
- `src/losses.py` AntiClass1MarginLoss
- `src/data_registry.py` DomainRegistry + DomainBalancedSampler


如果老师只允许提交一个 notebook，优先保证 `submission_notebook.ipynb` 能完整独立运行，并保留高分结果说明。
