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
