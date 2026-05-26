# 数字识别多专家异构系统 — 最终计划与成果

## 当前基线（TestA OOF）
- Strongest baseline: `testa_partial_init_lr1e4_mixup01_erasing005_e40`
  - OOF accuracy ≈ **0.7420**
  - class-1 predicted/true ratio ≈ **1.2746**
  - class-8 accuracy ≈ **0.6918**
  - total X→1 errors = **169**

## 为什么 TestA-only 不够
- 隐藏测试集分布未知，可能来自 MNIST/C、EMNIST、Kannada、Hasyv2、Chars74K、PenDigits 等。
- 同源 medium_cnn ensemble 提升微弱（误差模式高度相关）。
- 必须同时兼顾 TestA specialist + all-domain generalist + unknown-domain robust + anti-class-1 + 异构架构 + domain-aware ensemble。

## 多专家体系结构
| 专家家族 | 目的 | 代表配置 |
|---|---|---|
| TestA specialist (medium) | TestA 核心准确率 | `testa_medium_v2_raw_seed{2026,3407,777}_e60` |
| TestA anti-class-1 | 抑制 class-1 过预测 | `testa_medium_v2_anti1_margin_seed{42,2026,3407}_e60` |
| Heterogeneous (preact / convnext / convstem_vit) | 改变误差模式 | `testa_preact_resnet_tiny_v2_raw_seed42_e60` 等 |
| All-domain generalist | 已知域泛化 | `all_domain_medium_generalist_seed42_e80` 等 |
| Unknown-domain robust | 未知域鲁棒 | `all_domain_robust_light_seed42_e80` |

## 推荐运行顺序
1. TestA specialist (medium raw, multi-seed) → OOF eval + 诊断
2. TestA anti-class-1 (medium, multi-seed) → OOF eval
3. Heterogeneous experts → OOF eval
4. Multi-expert ensemble → grid search
5. All-domain generalist training + LODO evaluation
6. Domain-aware heuristic ensemble

## 单专家评估指标
每个专家至少报告：OOF accuracy / class1 ratio / class8 accuracy / total X→1 / best_epoch_mean / error_overlap_with_baseline

## Ensemble 成功标准
- OOF >= 0.7450
- class1 ratio <= 1.20
- class8 accuracy >= 0.71
- x_to_1 减少 >= 15
- worst-domain 不崩

## 最终提交模型选择
按 ensemble 权重保留 top 专家；淘汰 OOF < 0.72 且无法改善 ensemble 的专家。

## 保留 / 淘汰标准
- 保留：OOF >= 0.73 且 (class1 ratio < 1.25 或 class8 acc > 0.69 或 X->1 < 165)
- 保留：ensemble 中能拿到非零权重的专家
- 淘汰：large_cnn（partial init 不充分）+ OOF < 0.70 且所有偏差指标更差的

## 新数据集接入流程
1. 将新数据放入 `data/<domain_name>/`
2. 复制 `experiments/domain_template_medium_specialist.yaml`
3. 修改 `experiment_id` 和 `data.domains`

## 防 public benchmark overfitting
- anti-class-1 loss 在训练时生效（不依赖 test-time logit 后处理）
- 所有超参由 YAML 配置驱动，无 hard-coded 针对 TestA 的 hack
- LODO 评估确保模型不在已知域上偷懒
