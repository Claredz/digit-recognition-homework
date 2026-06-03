# 手写数字识别 — AI 导论课程作业

基于 PyTorch 的手写数字识别项目，包含标准 MNIST-like 基准和 TestA 真实世界域适应两个评估赛道。

## 最终结果

### 标准基准（MNIST / EMNIST / QMNIST）

| 模型 | MNIST | EMNIST | QMNIST |
|---|---|---|---|
| MediumCNN (clean) | 0.9973 | 0.9973 | 0.9973 |

### TestA 域适应（主攻方向）

| 模型 | OOF Accuracy | 参数量 | 说明 |
|---|---|---|---|
| 5-expert ensemble (wide+anti1) | **0.7891** | — | 最终提交 |
| WideResNetTiny (单模型) | 0.7797 | ~2.8M | 最强单专家 |
| PreActResNetTiny | 0.7569 | ~1.3M | 辅助专家 |
| MediumCNN (specialist) | 0.7474 | ~0.5M | 稳健基线 |

相对基线 (+0.0471)，class1 过预测从 1.275 降至 1.130，class8 准确率从 0.692 升至 0.764。

### 标准基准 vs TestA 对比

| 模型 | 标准基准平均 | TestA OOF |
|---|---|---|
| MediumCNN (clean) | **0.9973** | — |
| MediumCNN (TestA specialist) | 0.9938 | 0.7474 |
| WideResNetTiny (TestA specialist) | 0.9812 | **0.7797** |

> MediumCNN specialist 在通用数据上几乎无退化（-0.35%），WideResNet 有明显 specialization trade-off（-1.6%）。

---

## 项目结构

```
├── src/                              # 核心模块
│   ├── config.py                     # ExperimentConfig 数据类
│   ├── data.py                       # MNIST / EMNIST / USPS / QMNIST / folder 多源数据加载
│   ├── model.py                      # SmallCNN / MediumCNN / LargeCNN + build_model() 工厂
│   ├── models/heterogeneous.py       # PreActResNet / WideResNet / ConvNeXt / ViT / MobileNetV3
│   ├── engine.py                     # 训练循环 (fit, run_epoch)，AMP / compile 支持
│   ├── losses.py                     # AntiClass1MarginLoss（抑制 class-1 过预测）
│   ├── data_registry.py              # DomainRegistry + DomainBalancedSampler
│   ├── preprocess.py                 # 图片预处理（自动反转、裁剪、居中）
│   ├── robust_data.py                # MNIST-C 腐蚀增强
│   ├── testa_robust_train.py         # TestA 5-fold specialist 训练主模块
│   ├── evaluate.py                   # 评估：validation / holdout / MNIST-C
│   ├── predict.py                    # 无标签图片预测 + TTA + CSV 导出
│   ├── error_analysis.py             # 错误分析：混淆矩阵 / X→1 诊断
│   ├── ensemble/domain_router.py     # HeuristicDomainRouter（JS 散度路由）
│   └── experiment_config.py          # YAML 实验配置加载
├── scripts/                          # 实验入口脚本
│   ├── train_all_domain.py           # 多域 generalist 训练
│   ├── run_testa_finetune_kfold.py   # TestA specialist 5-fold fine-tune
│   ├── eval_testa_specialist_oof.py  # specialist OOF 评估 + 诊断
│   ├── eval_testa_expert_ensemble_oof.py  # ensemble 权重网格搜索
│   └── eval_experts_on_standard_benchmarks.py  # 专家标准基准评估
├── experiments/                      # YAML 实验配置
├── tests/                            # pytest 测试
├── outputs_submission/               # 高分结果归档（只读，勿覆盖）
├── outputs_runs/                     # 实验输出
├── submission_notebook.ipynb         # 独立提交版 notebook
├── project_notebook.ipynb            # 工程化分步复现 notebook
└── teaching_notebook.ipynb           # 教学解释版 notebook
```

---

## 环境准备

Python 3.10+，CUDA 可选。

```bash
pip install -r requirements.txt
```

## 快速开始

### 训练

```bash
# 快速 baseline（1 epoch smoke test）
python -m src.train --project-root . --dataset-name mnist --model-name medium_cnn \
  --epochs 1 --batch-size 64 --run-name smoke

# 多源训练
python -m src.train --project-root . --dataset-name multisource --model-name medium_cnn \
  --use-emnist-digits --use-usps --use-qmnist --optimizer-type AdamW \
  --scheduler-type CosineAnnealingLR --run-name multisource

# TestA specialist 5-fold fine-tune
python scripts/run_testa_finetune_kfold.py --config experiments/testa_finetune_from_generalist.yaml

# 多域 generalist 训练
python scripts/train_all_domain.py --config experiments/all_domain_medium_generalist_seed42_e80.yaml
```

### 评估

```bash
# 评估 checkpoint 在标准 holdout 上
python -m src.evaluate --project-root . \
  --checkpoint outputs_submission/checkpoints/best_model_state.pt \
  --dataset-name mnist --model-name medium_cnn \
  --holdouts mnist_test emnist_digits_test qmnist_test10k

# TestA specialist OOF 评估
python scripts/eval_testa_specialist_oof.py --config experiments/testa_specialist_5fold.yaml

# 专家标准基准评估
python scripts/eval_experts_on_standard_benchmarks.py
```

### 预测

```bash
# 单模型预测
python -m src.predict --project-root . \
  --checkpoint outputs_submission/checkpoints/best_model_state.pt \
  --image-dir <图片目录> --model-name medium_cnn --use-tta

# 融合预测（specialist + generalist 加权）
python scripts/predict_testa_fusion.py \
  --config experiments/specialist_generalist_ensemble.yaml \
  --image-dir <无标签图片目录>
```

### 测试

```bash
python -m pytest
```

---

## 模型体系

### CNN 家族

| 模型 | 规模 | 用途 |
|---|---|---|
| `SmallCNN` | 轻量 | 教学、smoke test |
| `MediumCNN` | ~0.5M | 主干模型，通用+TestA specialist |
| `LargeCNN` | ~3M | 已淘汰 |

### 异构模型

| 模型 | 参数量 | 设计 | TestA OOF |
|---|---|---|---|
| `WideResNetTiny` | ~2.8M | PreAct block + 宽通道(48→96→192) | **0.7797** |
| `PreActResNetTiny` | ~1.3M | PreActivation ResNet | 0.7569 |
| `ConvNeXtMicro` | ~0.8M | Depthwise conv + LayerNorm | 已淘汰 |
| `ConvStemViT` | ~1.1M | CNN stem + Transformer | 已淘汰 |
| `MobileNetV3_28` | ~1.5M | Inverted residual + SE | 已淘汰 |

> 核心发现：加宽通道比加深层更有效；WideResNet 是唯一 scratch 训练即超过基线的架构。

模型名通过 `normalize_model_name()` 解析别名（如 `"medium"` → `"medium_cnn"`, `"vit"` → `"convstem_vit"`），然后 `build_model()` 分发构造。

---

## 关键设计

### TestA 训练策略

- **5-Fold 交叉验证**：保证 OOF 评估可信（每个样本由没见过它的 fold 预测）
- **Generalist → Specialist**：使用未见过 TestA 标签的 generalist checkpoint 初始化
- **保守增强**：默认关闭 CutMix / RandomErasing，raw image 为主视图
- **AntiClass1MarginLoss**：在 CE 基础上加 margin penalty 抑制 class-1 过预测

### Ensemble

```
P_ensemble = w1 × P_expert1 + w2 × P_expert2 + ...
```

最终配置：`wide_resnet(0.7) + medium_anti1_seed2026(0.2) + medium_raw_seed3407(0.1)`

`HeuristicDomainRouter` 基于 JS 散度和预测冲突做无学习的动态权重路由。

### 输出目录约定

- `outputs_submission/` — 高分结果归档，**只读，绝不覆盖**
- `outputs_runs/<run_name>/` — CLI `--run-name` 输出
- `outputs_runs/<experiment_id>/` — YAML 驱动实验输出
- `outputs/` — 默认回退输出

---

## 数据源

支持的训练数据：MNIST、EMNIST Digits、USPS、QMNIST、folder 格式自定义数据集。

EMNIST Digits 会单独应用方向修正，不影响其他数据源。Windows 下如果 DataLoader 卡住，设置 `--num-workers 0`。

## Notebook 使用

- `submission_notebook.ipynb` — 自包含提交版（交老师）
- `project_notebook.ipynb` — 工程入口，调用 `src/` 模块
- `teaching_notebook.ipynb` — 教学版，带详细解释

## 提交清单

通常需提交：`submission_notebook.ipynb`、`README.md`、预测 CSV（如老师要求）、必要截图或实验报告。
