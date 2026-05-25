# TestA 专家多样性下一阶段操作手册

## 当前现状

主线 5-Fold OOF accuracy 卡在 **0.7420**（`testa_partial_init_lr1e4_mixup01_erasing005_e40`），所有同源消融（lightmorph / weakaffine / weakaffine_weakblur / dilateheavy）都没突破。诊断显示模型存在**结构性 class-1 吸盘**：

- predicted_count / true_count 中，class 1 比例 ≈ 1.27（过预测），class 8 ≈ 0.87（被压制）
- 大量 8/9/6/5/3/2/4 被错预测为 1，总计 ~165 错样本
- logit-prior 校准实验证明这是"自信的错"，无法通过 inference-time 修正
- 6 个 medium_cnn variant 的 soft ensemble 也不能突破——错误模式高度相关

突破需要**真正的多样性**：不同 seed、不同 architecture、不同增广方向。

## 三个新专家配置

| 配置文件 | 关键差异点 | 用于打破什么 |
|---|---|---|
| `experiments/testa_medium_raw_seed2026.yaml` | seed=2026 + init=`robust_expert_best.pt` | 不同 fold split + 不同初始化随机性 |
| `experiments/testa_medium_raw_anti1_morph015.yaml` | morph_p=0.15 + dilate_bias=0.70 | 直接对抗 class-1 吸盘的轻量版增广 |
| `experiments/testa_large_raw_scratch.yaml` | large_cnn from scratch, 30 epoch | 不同 architecture、零 MNIST 归纳偏置 |

所有 3 个都使用 raw 预处理、`outputs_runs/<experiment_id>/`，绝不覆盖 `outputs_submission/`。

## 推荐运行顺序（cost vs 突破概率）

```bash
# 1. medium seed2026  (~6 min) — 最容易先尝试
python scripts/run_testa_finetune_kfold.py --config experiments/testa_medium_raw_seed2026.yaml
python scripts/eval_testa_specialist_oof.py --config experiments/testa_medium_raw_seed2026.yaml

# 2. anti-class-1 specialist  (~6 min)
python scripts/run_testa_finetune_kfold.py --config experiments/testa_medium_raw_anti1_morph015.yaml
python scripts/eval_testa_specialist_oof.py --config experiments/testa_medium_raw_anti1_morph015.yaml

# 3. large scratch  (~15 min)
python scripts/run_testa_finetune_kfold.py --config experiments/testa_large_raw_scratch.yaml
python scripts/eval_testa_specialist_oof.py --config experiments/testa_large_raw_scratch.yaml
```

## 4-专家 soft-vote 集成

```bash
python scripts/eval_testa_expert_ensemble_oof.py \
  --experiments testa_finetune_lr1e4_mixup01_erasing005_e40 \
                testa_medium_raw_seed2026 \
                testa_medium_raw_anti1_morph015 \
                testa_large_raw_scratch \
  --grid-search --grid-step 0.1
```

输出写入 `outputs_runs/expert_ensemble_analysis/`：

- `expert_ensemble_summary.json` — 全部权重候选 + 4 个 best（按 overall acc / class-1 ratio / class-8 acc / total X→1 各一个）
- `expert_ensemble_summary.csv` — 一行一个权重组合，含核心诊断列
- `best_ensemble_oof_predictions.csv` — best 权重下逐样本预测
- `best_ensemble_oof_probabilities.pt` — 可后续做 full-TestA 提交融合
- `confusion_matrix.csv` — best 权重下的混淆矩阵
- `per_expert_diagnostics.csv` — 每个单独专家的同套诊断指标（直接对照）

可选参数：

- `--grid-step 0.05` — 更细的网格（4 专家约 1771 个候选，仍然秒级）
- `--equal-weight-only` — 跳过 grid，只跑等权
- `--oof-paths PATH ...` — 直接给文件路径，绕过 experiment_id 解析

## 判断 ensemble 是否"真的"有效

**只看 overall accuracy 会被随机方差骗**。必须同时满足以下 4 条才算真正打破 class-1 偏差：

| 指标 | baseline (e40) | 阈值（必须同时达到） |
|---|---:|---:|
| overall OOF accuracy | 0.7420 | ≥ **0.7450** |
| class 1 predicted/true ratio | 1.274 | ≤ **1.20** |
| class 8 accuracy | 0.6918 | ≥ **0.71** |
| total X→1 errors | ~165 | 减少 ≥ **15** |

如果只 overall acc 涨 0.0005 而 class 1 ratio 不动，那只是同源失败的重演——**不要提交**。

## large_cnn OOF 低于 medium_cnn 怎么办

**仍然保留**为专家，前提是它在以下任一指标上明显更好：

- `class_8_accuracy ≥ 0.71`
- `total_x_to_1_errors` 比 medium 少 ≥ **10**

理由：集成的目标是**互补性**，不是每个专家都要最强。grid-search 会自动把弱专家权重压到合适位置，不会拖累整体。

**唯一例外**：若 large_cnn OOF ≤ 0.71 **且** class-1 ratio 与 medium 几乎相同，说明 scratch 训练根本没收敛。这时考虑加 epochs 重训，而不是把它扔进 ensemble。

## 安全约定

- 所有新结果写入 `outputs_runs/<experiment_id>/`
- 严禁覆盖 `outputs_submission/`
- ensemble 脚本默认输出到 `outputs_runs/expert_ensemble_analysis/`，并显式拒绝指向 `outputs_submission/`
- 缺失 OOF 时脚本会清晰报错并打印需要先运行的命令

---

## 实测结果（截至 2026-05-26）

### 单专家 OOF accuracy

| 实验 | init checkpoint | OOF | 备注 |
|---|---|---:|---|
| `testa_partial_init_lr1e4_mixup01_erasing005_e40` | v2_testa_partial (0.71) | **0.7420** | 当前主线 baseline |
| `testa_medium_raw_seed2026_e40` (40 epoch) | robust_expert_best (v1) | 0.7034 | 不同 seed/init，无 morph |
| `testa_medium_raw_anti1_morph015_e40` (40 epoch) | robust_expert_best (v1) | 0.7028 | dilate-biased morph_p=0.15 |
| `testa_large_raw_scratch_e60` (60 epoch scratch) | none | 0.6822 | large_cnn from scratch |

### 关键发现：epoch 不够是真问题

3 个新专家的 e20/e30 第一版几乎全部 best_epoch 都在训练后期，e40/e60 重训后单专家 OOF 提升 +0.012 ~ +0.060。所以"更长 epoch"这一步必须做，否则 5-fold 报告的 best_val_accuracy 是被早停截断的下界。

### 4-专家 ensemble 结果

`testa_partial_init_lr1e4_mixup01_erasing005_e40 + seed2026_e40 + anti1_e40 + large_e60`，grid step 0.1，286 个权重候选：

| 候选 | overall | c1_ratio | c8_acc | X→1 |
|---|---:|---:|---:|---:|
| baseline (e40 alone) | 0.7420 | 1.275 | 0.6918 | 169 |
| **best overall** `[0.3, 0.0, 0.3, 0.4]` | **0.7426** | 1.285 | 0.6858 | 166 |
| match baseline + 减 bias `[0.3, 0.0, 0.4, 0.3]` | 0.7420 | 1.262 | 0.6828 | 162 |
| best c8 acc `[0.3, 0.0, 0.1, 0.6]` | 0.7374 | 1.272 | 0.6918 | 162 |
| **best X→1** `[0.1, 0.1, 0.5, 0.3]` | 0.7317 | **1.233** | 0.6707 | **155** |

### 严格阈值评估

文档里设的 4 条同时满足阈值：

| 指标 | baseline | 阈值 | 最强候选实际 |
|---|---:|---:|---:|
| overall acc | 0.7420 | ≥ 0.7450 | 0.7426 ❌ |
| class 1 ratio | 1.274 | ≤ 1.20 | 1.200 ❌（独立达成）|
| class 8 acc | 0.6918 | ≥ 0.71 | 0.6918 ❌（仅持平） |
| total X→1 | 169 | ≤ 154 | 155 ❌（差 1）|

**没有任何组合同时满足全部 4 条**。所以这一轮的结论是：

1. **微弱有效**：best overall 比 baseline 高 0.0006（接近随机方差水平）
2. **bias 方向可调**：可以用权重换 0.0103 overall accuracy 来减 14 个 X→1 错误（`[0.1, 0.1, 0.5, 0.3]`），这在追求 class 8/9 等"被压制类"上是有意义的工程权衡
3. **没有清晰的全面胜利**——init checkpoint 决定了 ceiling，3 个新专家从 v1 init 起步，比 e40 用的 v2 partial init 弱 0.04 是结构性的

### 下一步建议（按 ROI）

1. **同 init、多 seed**：从 `robust_expert_v2_testa_partial_best_epoch12_score07098.pt` 出发，训 seed=2026 / 3407 的 medium_cnn——这才是真正测试"seed 多样性"的实验，预期单专家 ~0.74，集成可能突破 0.745
2. **保留 large_cnn 实验**：单独 0.6822 不够强，但 weight=0.3-0.6 在 best overall 和 c8 acc 维度都被选中——它学到的特征跟 medium 不一样
3. **当前不应该提交 ensemble**：strict 阈值未达，按文档约定不算 real win

