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

如果老师只允许提交一个 notebook，优先保证 `submission_notebook.ipynb` 能完整独立运行，并保留高分结果说明。
