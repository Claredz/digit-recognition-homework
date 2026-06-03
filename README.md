# 手写数字识别 — AI 导论课程作业

基于 PyTorch 的手写数字识别，最终提交为 **5 专家 Dynamic MoE 集成系统**。

## 最终提交

**`build/钟兴涛-25120617.zip`**（38.52 MB，18 文件），默认 `--router dynamic`。

## 5 专家阵容

| # | 专家 | 架构 | 新 TestA | 标准域 | 角色 |
|---|---|---|---|---|---|
| 1 | `wide_resnet_tiny_raw_seed42` | WideResNetTiny (~2.8M) | 92.43% | 98.36% | TestA 域专家，与 robust_v1 互补 |
| 2 | `medium_anti1_seed2026` | MediumCNN | 93.26% | 99.34% | 抑制 class-1 过度预测 |
| 3 | `medium_raw_seed3407` | MediumCNN | 93.23% | 99.34% | TestA 域专家，稳定基线 |
| 4 | `robust_v1` | MediumCNN | 92.80% | — | 鲁棒增强训练，错误模式独特 |
| 5 | `MNIST_clean` | MediumCNN | 76.57% | ~99.7% | 标准域锚点，拉高 MNIST-like 准确率 |

> 专家 1-3 为 5-fold 交叉验证训练，提交时 5 折概率取均值。专家 4-5 为单 checkpoint。
> 新 TestA 为教师发布测试集（3500 样本，TTA=1），标准域为 MNIST-family 均值。

### 为什么选这 5 个？

核心原则：**错误互补性 > 单体准确率**。

| 配对 | 错误重叠率 | 集成上限 |
|---|---|---|
| wide_resnet + robust_v1 | **47%** | **96.46%** |
| wide_resnet + MNIST_clean | 60% | 95.43% |
| medium_anti1 + medium_raw | **95%** | 93.60% |

> MediumCNN 同架构不同种子间错误重叠 88-97%，堆再多也不互补。WideResNet（大架构）+ robust_v1（鲁棒训练）+ MNIST_clean（纯净训练）来自完全不同的训练策略，是互补性的来源。

## Router 对比

### 新 TestA（教师发布版，TTA=8）

| Router | 准确率 | 说明 |
|---|---|---|
| **rule** | **94.60%** | 域感知规则路由 |
| static | 94.37% | 固定权重 [0.40, 0.10, 0.30, 0.15, 0.05] |
| dynamic | 94.29% | 逐样本特征门控（**默认**） |

### 标准域（TTA=1）

| 域 | 样本数 | static | dynamic | rule | average |
|---|---|---|---|---|---|
| MNIST test | 10,000 | 99.48% | 99.59% | 99.54% | **99.60%** |
| QMNIST test10k | 10,000 | 99.48% | 99.59% | 99.54% | **99.60%** |
| EMNIST digits | 10,000 | 99.57% | **99.68%** | 99.67% | 99.67% |
| USPS train | 7,291 | 97.01% | 97.19% | 97.15% | **97.26%** |

> 标准域上 average 最优或接近最优，dynamic 在 EMNIST 上略优。TestA 域偏移场景下 dynamic/rule 路由价值更大。

## 动态权重系统

### Dynamic MoE Router（默认）

逐样本计算 5 专家权重，无需任何标签：

```
score[e] = log(base_w[e]) + λ₁×conf[e] + λ₂×margin[e] − λ₃×disagree[e] + λ₄×anti1[e] + clean_boost[e] + robust_boost[e]
weights = softmax(score)
```

6 个 per-expert 特征全自动从预测概率中提取：
- **conf[e]**：置信度（中心化）
- **margin[e]**：top-1 − top-2 间隔（中心化）
- **disagree[e]**：专家 e 与加权平均分布的 KL 散度（惩罚与众不同的专家）
- **anti1[e]**：当集成预测为 class-1 时提升 anti1 专家（抑制误判）
- **clean_boost[e]** / **robust_boost[e]**：特定专家的置信度奖励

### Rule Router（备选）

基于规则判断样本域归属，套用预设模板：
- `balanced`（默认）：[0.30, 0.15, 0.25, 0.20, 0.10]
- `testa`：提升 WideResNet + anti1 → [0.55, 0.15, 0.20, 0.08, 0.02]
- `mnist`：提升 MNIST_clean → [0.20, 0.10, 0.25, 0.15, 0.30]
- `robust`：提升 robust_v1 → [0.25, 0.10, 0.20, 0.40, 0.05]

## 提交包

```bash
# 教师系统运行命令（默认 dynamic router）
python3 predict.py --testdata /testdata --output /results/submission.csv

# 可选 router
python3 predict.py --testdata /testdata --output /results/submission.csv --router rule
```

`predict.py` 自包含，无 `src/` 依赖，内联所有模型定义（MediumCNN, LargeCNN, PreActResNet, PreActResNetTiny）。

## 项目结构

```
├── src/                          # 核心模块（config, data, model, engine, losses, ...）
├── scripts/                      # 实验脚本
│   ├── build_5expert_dynamic_moe_submission.py
│   ├── moe_expert_selection_pipeline.py
│   └── eval_5expert_other_domains.py
├── predict.py                    # 自包含提交脚本
├── experiments/                  # YAML 实验配置
├── tests/                        # pytest
├── outputs_submission/           # 高分归档（只读）
├── outputs_runs/                 # 实验输出
└── build/
    └── 钟兴涛-25120617.zip       # 最终提交包
```

## 快速开始

```bash
pip install -r requirements.txt
python -m src.train --project-root . --dataset-name mnist --model-name medium_cnn --epochs 1
python -m pytest
```

## 提交清单

- `build/钟兴涛-25120617.zip`（predict.py + 17 checkpoint）
- `README.md`
- 预测 CSV
