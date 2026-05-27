# 手写数字识别作业提交说明

## 主要提交文件

- `submission_notebook.ipynb`：提交用 notebook。
- `predictions/submission_kfold_testA.csv`：最终 TestA 预测结果。
- `predictions/submission_kfold_testA.manifest.json`：预测文件生成配置与模型清单。
- `checkpoints/robust_expert_v2_kfold_f0_best.pt` 至 `robust_expert_v2_kfold_f4_best.pt`：5 折鲁棒微调模型 checkpoint。

## 评估与结果文件

- `evaluation/testA_kfold_ensemble.json`：TestA k-fold ensemble 评估结果。
- `evaluation/testa_kfold_summary.json`：5 折训练/验证摘要。
- `evaluation/final_weighted_ensemble_four_domain_eval.json`：四个域上的最终加权 ensemble 评估。
- `evaluation/final_weighted_ensemble_standard_benchmarks.json`：MNIST / EMNIST Digits / QMNIST 标准 benchmark 评估。
- `figures/confusion_matrix.png`：混淆矩阵图。
- `figures/misclassified_examples.png`：错误样例图。

## 关键结果摘要

- TestA 最终固定加权 ensemble OOF：0.7891。
- MNIST-family：0.9896。
- MNIST-C：0.9496。
- local/external digits：0.5846。

## 备注

本目录是从项目最终归档输出 `outputs_submission/` 和少量 `outputs_runs/` 评估摘要整理得到的提交包。原始训练数据和大型中间缓存没有包含在提交包中。
