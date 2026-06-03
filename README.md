# 手写数字识别 — AI 导论课程作业

基于 PyTorch 的手写数字识别项目，包含标准 MNIST-like 基准和 TestA 真实世界域适应两个评估赛道。

## 最终结果

### 标准基准

| 模型 | MNIST | EMNIST | QMNIST |
|---|---|---|---|
| MediumCNN (clean) | 0.9973 | 0.9973 | 0.9973 |

### TestA（教师发布版，3500 样本，10 类均衡）

| 策略 | 权重 | 准确率 |
|---|---|---|
| **5专家混合最佳静态** | 随机搜索 | **94.83%** |
| 3专家静态网格搜索 | wide=0.40, anti1=0.10, raw=0.50 | 94.14% |
| 3专家 mnist 模板 | 0.35/0.35/0.30 | 94.06% |
| 3专家固定权重（训练最优） | 0.70/0.20/0.10 | 93.40% |
| Dynamic MoE（逐样本） | conf=5.0, margin=1.5 | 93.94% |
| Domain-Aware Rule Router | 域感知规则 | 94.06% |

> 训练分布（旧 TestA OOF）上最优的 0.7/0.2/0.1 在新 TestA 上不是最优——分布有偏移，新 TestA 更接近标准 MNIST。

### 训练分布 TestA OOF（5-fold 交叉验证，3500 样本，类不均衡）

| 模型 | OOF Accuracy | 说明 |
|---|---|---|
| 3-expert ensemble (0.7/0.2/0.1) | **0.7891** | 最终提交配置 |
| WideResNetTiny | 0.7797 | 最强单专家 |
| PreActResNetTiny (anti1) | 0.7569 | |
| PreActResNetTiny (raw) | 0.7514 | |
| MediumCNN v2 anti1 ×3 种子 | 0.7451–0.7477 | |
| MediumCNN v2 raw ×3 种子 | 0.7449–0.7471 | |
| ConvNeXtMicro | 0.6129 | 已淘汰 |
| ConvStemViT | 0.6017 | 已淘汰 |
| MobileNetV3_28 | 0.4877 | 已淘汰 |

### 多域评估

| 域 | 样本数 | 3专家固定 (0.7/0.2/0.1) | 3专家 mnist 模板 |
|---|---|---|---|
| MNIST-family | 67,291 | 98.96% | 99.34% |
| MNIST-C | 160,000 | 94.96% | 96.59% |
| local/external digits | 34,546 | 58.46% | 62.65% |

---

## 动态权重系统

### 1. Dynamic MoE Router
逐样本线性门控：`score[e] = log(base_w[e]) + λ₁×confidence[e] + λ₂×margin[e] − λ₃×disagreement[e] + λ₄×anti1[e]`，softmax 后得权重。4 个特征（置信度、间隔、分歧、anti1 信号）per-expert 自动计算，无需标签。

### 2. Domain-Aware Rule Router
4 条 if-else 规则判断样本"像哪个域"，然后套用该域的预设权重模板（testa / mnist / balanced / anti1）。

### 3. Learned MoE Router
训练小型 NN（30→3 线性门），输入 3 个专家概率拼接。多域加权目标优化。

### 关键发现
- **专家多样性 >> 路由复杂度**：MediumCNN 不同种子间错误重叠 88–97%，加再多也不互补。真正互补的是架构差异大的专家。
- **WideResNet + robust_v1 错误重叠仅 47%**，是最互补的配对。
- **robust_v1（数据增强训练的 MediumCNN）在新 TestA 上单模型 92.80%**，接近 TestA 专家。

---

## 新 TestA 专家排名（教师发布版）

| 排名 | 专家 | 准确率 | vs 旧 TestA OOF |
|---|---|---|---|
| 1 | medium_v2_raw_seed777 | 93.31% | +18.8% |
| 2 | medium_anti1_seed2026 | 93.26% | +18.8% |
| 3 | medium_raw_seed3407 | 93.23% | +18.5% |
| 4 | partial_init_mixup | 93.06% | +18.9% |
| 5 | robust_v1 | 92.80% | — |
| 6 | wide_resnet_tiny | 92.43% | +14.5% |
| 7 | robust_kfold_ens | 91.97% | — |
| 8 | preact_resnet_anti1 | 91.26% | +15.6% |
| 9 | preact_resnet_raw | 91.11% | +16.0% |
| 10 | large_cnn_v2 | 86.94% | +18.7% |
| — | MNIST_clean | 76.57% | — |

> **排名完全反转**：旧 TestA 上 WideResNet (78%) > MediumCNN (74%)；新 TestA 上 MediumCNN (93.3%) > WideResNet (92.4%)。分布偏移是主导因素。

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

见 `build/submission.zip`（36.7 MB），包含自包含的 `predict.py` + 15 个 checkpoint。

```bash
# 教师系统运行命令
python3 predict.py --testdata /testdata --output /results/submission.csv
```

`predict.py` 内联所有模型定义，无 `src/` 依赖，仅使用允许的库。

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
│   ├── predict_final_weighted_ensemble.py  # 3专家集成入口
│   ├── search_dynamic_moe_router_oof.py    # Dynamic MoE 搜索
│   ├── search_domain_aware_rule_router.py  # 域感知路由搜索
│   ├── train_learned_moe_router.py         # 学习式路由训练
│   ├── eval_experts_new_testA.py           # 新 TestA 专家评估
│   ├── eval_routing_testA.py               # 路由策略评估
│   ├── eval_clean_robust_testA.py          # clean/robust 模型评估
│   └── eval_hybrid_ensemble.py             # 混合集成评估
├── predict.py                        # 自包含提交脚本（无 src 依赖）
├── experiments/                      # YAML 实验配置
├── tests/                            # pytest
├── outputs_submission/               # 高分归档（只读）
├── outputs_runs/                     # 实验输出（29 个 TestA 专家 + 动态系统）
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

需提交：`build/submission.zip`（predict.py + 15×checkpoint）、`README.md`、预测 CSV。
