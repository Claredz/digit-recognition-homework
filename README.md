# 手写数字识别 — AI 导论课程作业

基于 PyTorch 的手写数字识别项目，包含标准 MNIST-like 基准和 TestA 真实世界域适应两个评估赛道。

## 最终提交

**`build/钟兴涛-25120617.zip`**（38.52 MB，18 文件），默认 `--router dynamic` 动态 MoE 路由。

### 5 专家阵容

| # | 专家 | 架构 | 类型 | 说明 |
|---|---|---|---|---|
| 1 | `wide_resnet_tiny_raw_seed42` | WideResNetTiny (~2.8M) | 5-fold TestA 专家 | 旧 TestA 最强单体，与 robust_v1 高度互补 |
| 2 | `medium_anti1_seed2026` | MediumCNN | 5-fold anti1 专家 | 抑制 class-1 过度预测 |
| 3 | `medium_raw_seed3407` | MediumCNN | 5-fold TestA 专家 | 标准训练，稳定基线 |
| 4 | `robust_v1` | MediumCNN | 单 checkpoint | 数据增强鲁棒训练，错误模式独特 |
| 5 | `MNIST_clean` | MediumCNN | 单 checkpoint | MNIST 纯净训练，标准域锚点 |

### Router 对比：新 TestA（教师发布版，TTA=8）

| Router | 准确率 | 说明 |
|---|---|---|
| **rule** | **94.60%** | 域感知规则（4 模板），最佳观测 |
| static | 94.37% | 固定权重 [0.40, 0.10, 0.30, 0.15, 0.05] |
| dynamic | 94.29% | 逐样本特征门控（默认） |
| 3-expert 固定 | 93.40% | 旧最优 0.70/0.20/0.10 |

### Router 对比：标准域（TTA=1）

| 域 | 样本数 | static | dynamic | rule | average | 最优 |
|---|---|---|---|---|---|---|
| MNIST test | 10,000 | 99.48% | 99.59% | 99.54% | **99.60%** | average |
| QMNIST test10k | 10,000 | 99.47% | 99.59% | 99.54% | **99.60%** | average |
| EMNIST digits | 10,000 | 99.57% | **99.68%** | 99.67% | 99.67% | dynamic |
| USPS train | 7,291 | 97.01% | 97.19% | 97.15% | **97.26%** | average |

> **观察**：标准域上简单 average 通常最优或接近最优；TestA 域偏移场景下 dynamic/rule 路由更有价值。

### 训练分布 TestA OOF（5-fold 交叉验证，3500 样本，类不均衡）

| 模型 | OOF Accuracy | 说明 |
|---|---|---|
| 3-expert ensemble (0.7/0.2/0.1) | **0.7891** | 原始最终提交 |
| WideResNetTiny | 0.7797 | 最强单专家 |
| PreActResNetTiny (anti1) | 0.7569 | |
| PreActResNetTiny (raw) | 0.7514 | |
| MediumCNN v2 anti1 ×3 种子 | 0.7451–0.7477 | |
| MediumCNN v2 raw ×3 种子 | 0.7449–0.7471 | |

### 多域评估（3 专家固定权重 0.7/0.2/0.1）

| 域 | 样本数 | 准确率 |
|---|---|---|
| MNIST-family | 67,291 | 98.96% |
| MNIST-C | 160,000 | 94.96% |
| local/external digits | 34,546 | 58.46% |

---

## 动态权重系统（5 专家版）

### 1. Dynamic MoE Router（默认）
逐样本特征门控，支持 N 专家。公式：
`score[e] = log(base_w[e]) + λ₁×confidence[e] + λ₂×margin[e] − λ₃×disagreement[e] + λ₄×anti1[e] + clean_boost[e] + robust_boost[e]`

6 个 per-expert 特征：
- **confidence[e]**：专家 e 的 top-1 概率（中心化）
- **margin[e]**：top-1 − top-2 间隔（中心化）
- **disagreement[e]**：专家 e 与加权平均分布的 KL 散度
- **anti1[e]**：当加权预测为 class-1 时提升 anti1 专家权重
- **clean_boost[e]** / **robust_boost[e]**：特定专家的置信度奖励信号

softmax 归一化后得逐样本权重，全自动，无需任何标签。

### 2. Domain-Aware Rule Router
4 条 if-else 规则判断样本"像哪个域"，套用预设权重模板：
- `testa`：TestA 专家主导 [0.55, 0.15, 0.20, 0.08, 0.02]
- `mnist`：clean 专家主导 [0.20, 0.10, 0.25, 0.15, 0.30]
- `robust`：robust 专家提升 [0.25, 0.10, 0.20, 0.40, 0.05]
- `anti1`：anti1 专家提升 [0.35, 0.35, 0.15, 0.10, 0.05]
- `balanced`：默认均衡 [0.30, 0.15, 0.25, 0.20, 0.10]

### 3. Static / Average
固定权重或等权重平均，作为无参数基线。

### 关键发现
- **专家多样性 >> 路由复杂度**：MediumCNN 不同种子间错误重叠 88–97%，加再多也不互补。真正互补的是训练策略/架构差异大的专家。
- **WideResNet + robust_v1 错误重叠仅 47%**，是最互补的配对（集成上限 96.46%）。
- **标准域 vs TestA 的最优策略不同**：标准域上 simple average 最优；TestA 上 dynamic/rule 路由胜出。
- **分布偏移**：旧 TestA OOF (WideResNet 78% > MediumCNN 74%) vs 新 TestA (MediumCNN 93.3% > WideResNet 92.4%)，排名完全反转。

---

## 新 TestA 专家排名（教师发布版，3500 样本）

训练分布 (旧 TestA OOF) vs 新 TestA 的排名完全反转，说明两个分布存在实质性差异。

| 排名 | 专家 | 新 TestA | vs 旧 TestA OOF | 架构 |
|---|---|---|---|---|
| 1 | medium_v2_raw_seed777 | 93.31% | +18.8% | MediumCNN |
| 2 | medium_anti1_seed2026 | 93.26% | +18.8% | MediumCNN |
| 3 | medium_raw_seed3407 | 93.23% | +18.5% | MediumCNN |
| 4 | partial_init_mixup | 93.06% | +18.9% | MediumCNN |
| 5 | robust_v1 | 92.80% | — | MediumCNN |
| 6 | wide_resnet_tiny | 92.43% | +14.5% | WideResNetTiny |
| 7 | robust_kfold_ens | 91.97% | — | MediumCNN×5 |
| 8 | preact_resnet_anti1 | 91.26% | +15.6% | PreActResNet |
| 9 | preact_resnet_raw | 91.11% | +16.0% | PreActResNet |
| 10 | large_cnn_v2 | 86.94% | +18.7% | LargeCNN |
| — | MNIST_clean | 76.57% | — | MediumCNN |

> MediumCNN 在新 TestA 上超越 WideResNet (+0.9%)，旧 TestA 上 WideResNet 领先 (+3.5%)。

---

## 混合专家互补性

| 配对 | 错误重叠率 | 集成上限 |
|---|---|---|
| wide_resnet + robust_v1 | **47%** | **96.46%** |
| wide_resnet + MNIST_clean | 60% | 95.43% |
| robust_v1 + MNIST_clean | 84% | 93.97% |
| medium_anti1 + medium_raw | **95%** | 93.60% |

> 不同训练策略的专家互补性远大于同架构不同种子。

---

## 提交包

见 **`build/钟兴涛-25120617.zip`**（38.52 MB，18 文件），5 专家 dynamic MoE。

```bash
# 教师系统运行命令（默认 dynamic router）
python3 predict.py --testdata /testdata --output /results/submission.csv

# 可选 router
python3 predict.py --testdata /testdata --output /results/submission.csv --router rule
python3 predict.py --testdata /testdata --output /results/submission.csv --router static
```

`predict.py` 内联所有模型定义（MediumCNN, LargeCNN, PreActResNet, PreActResNetTiny），无 `src/` 依赖，仅使用 `torch`, `torchvision`, `numpy`, `argparse`, `sys`, `pathlib`, `struct`。

---

## 项目结构

```
├── src/                              # 核心模块
│   ├── config.py                     # ExperimentConfig 数据类
│   ├── data.py                       # MNIST / EMNIST / USPS / QMNIST 多源数据
│   ├── model.py                      # SmallCNN / MediumCNN / LargeCNN
│   ├── models/heterogeneous.py       # PreActResNet / WideResNet / ConvNeXt / ViT
│   ├── engine.py                     # 训练循环，AMP / compile
│   ├── losses.py                     # AntiClass1MarginLoss
│   ├── data_registry.py              # DomainRegistry
│   ├── preprocess.py                 # 图片预处理
│   ├── evaluate.py                   # 评估
│   ├── predict.py                    # 预测 + TTA
│   └── ensemble/domain_router.py     # HeuristicDomainRouter
├── scripts/                          # 实验脚本
│   ├── build_5expert_dynamic_moe_submission.py  # 5 专家 MoE 提交包构建
│   ├── moe_expert_selection_pipeline.py     # 专家扫描→互补性分析→路由搜索
│   ├── eval_experts_new_testA.py           # 新 TestA 专家评估
│   ├── eval_5expert_other_domains.py       # 5 专家标准域评估
│   ├── search_dynamic_moe_router_oof.py    # Dynamic MoE 参数搜索
│   ├── predict_testa_fusion.py             # 专家+通用融合预测
│   ├── eval_testa_expert_ensemble_oof.py   # 集成权重网格搜索
│   └── eval_testa_specialist_oof.py        # 专家 OOF 诊断
├── predict.py                        # 自包含提交脚本（无 src 依赖）
├── experiments/                      # YAML 实验配置
├── tests/                            # pytest
├── outputs_submission/               # 高分归档（只读）
├── outputs_runs/                     # 实验输出
│   ├── moe_expert_selection/         # 专家筛选结果
│   └── moe_dynamic_router/           # 动态路由评估
├── build/
│   └── 钟兴涛-25120617.zip           # 最终提交包
└── exam_final_archive_2026-06-02/    # 考前归档
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 训练
python -m src.train --project-root . --dataset-name mnist --model-name medium_cnn --epochs 1

# 评估
python -m src.evaluate --project-root . --checkpoint <path.pt> --dataset-name mnist --model-name medium_cnn

# 预测
python -m src.predict --project-root . --checkpoint <path.pt> --image-dir <dir> --use-tta

# 测试
python -m pytest
```

## 提交清单

需提交：`build/钟兴涛-25120617.zip`（predict.py + 17 checkpoint）、`README.md`、预测 CSV。
