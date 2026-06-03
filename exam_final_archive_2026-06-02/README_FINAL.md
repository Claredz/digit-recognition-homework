# 考试前最终结果归档 - 2026-06-02

本目录保存明天正式测试前的最终状态。归档包含固定权重最终提交系统、动态调权系统、15 个最终 checkpoint、关键结果、预测代码和运行命令。

## 最终推荐提交系统

默认推荐使用 **fixed_weight_final**：

| Expert | Model | Weight | Folds |
|---|---|---:|---:|
| wide_resnet_tiny_raw_seed42 | wide_resnet_tiny | 0.7 | 5 |
| medium_anti1_seed2026 | medium_cnn | 0.2 | 5 |
| medium_raw_seed3407 | medium_cnn | 0.1 | 5 |

固定权重公式：`P_final = 0.7 * P_wide + 0.2 * P_anti1 + 0.1 * P_raw`，其中每个专家先对 5 个 fold 的概率取平均。

## 最终关键结果

| Dataset / Domain | Accuracy |
|---|---:|
| TestA OOF | 0.7891 |
| MNIST test | 0.9902 |
| EMNIST Digits test | 0.9901 |
| QMNIST test10k | 0.9902 |
| MNIST-family | 0.9896 |
| MNIST-C | 0.9496 |
| local/external digits | 0.5846 |

低分风险集：`hasyv2_digits 0.3912`、`penbased_rendered 0.5886`、`canny_edges 0.6879`、`zigzag 0.8131`。

## 动态调权系统

动态系统也已归档在 `results/dynamic_weight_systems/`：

- `dynamic_moe_router/`：基于 confidence、margin、disagreement 等特征搜索动态 MoE 权重。OOF in-sample 最好约 `0.7903`，但 CV 约 `0.7854`，所以不作为 TestA 默认提交。
- `domain_aware_rule_router/`：面向混合隐藏域的规则路由。相对固定 TestA 权重，MNIST-family / MNIST-C / local-external 更好，但 TestA 降到约 `0.7834`。
- `learned_moe_router/`：学习式 router 实验结果，作为备份和分析依据保存。

结论：如果正式测试明确接近 TestA，使用固定权重最终系统；如果老师临时说明隐藏集是多域混合，可参考动态调权结果。

## 明天正式运行

优先使用：`C:\Users\claredz\anaconda3\python.exe`。

图片文件夹模式：

```bat
cd /d "E:\ALL\学习\AI导论作业-识别手写数字"
C:\Users\claredz\anaconda3\python.exe scripts\predict_final_weighted_ensemble.py ^
  --image-dir "<老师给的图片文件夹路径>" ^
  --output "outputs_runs\final_weighted_ensemble_predictions\submission_final_weighted.csv" ^
  --tta-n 8 ^
  --batch-size 256
```

IDX 文件模式：

```bat
cd /d "E:\ALL\学习\AI导论作业-识别手写数字"
C:\Users\claredz\anaconda3\python.exe scripts\predict_final_weighted_ensemble.py ^
  --idx-images "<老师给的idx图像文件路径>" ^
  --output "outputs_runs\final_weighted_ensemble_predictions\submission_final_weighted.csv" ^
  --tta-n 8 ^
  --batch-size 256
```

提交文件：`outputs_runs\final_weighted_ensemble_predictions\submission_final_weighted.csv`。

不要直接提交旧文件：`outputs_submission\predictions\submission_kfold_testA.csv`。它指向旧 K 折模型，不是最终 3 专家固定权重系统。

## 目录说明

- `models/`：15 个最终 checkpoint。
- `code/`：最终预测脚本、`src/` 源码、关键实验 YAML。
- `results/fixed_weight_final/`：最终固定权重系统结果。
- `results/dynamic_weight_systems/`：动态调权系统结果与缓存。
- `predictions/`：烟测预测输出。
- `run_commands/`：明天可直接参考的 Windows 命令。
- `FINAL_MANIFEST.json`：机器可读清单。
