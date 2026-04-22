# 手写数字识别作业（基线工程）

这是一个面向学习的手写数字识别基线工程，用于 AI 导论课程。

## 这个仓库能做什么

- 训练一个小型 CNN 手写数字分类器
- 保存 checkpoint、训练日志与学习曲线
- 生成混淆矩阵与误分类样例图
- 对无标签图片批量预测并导出 CSV（便于最后提交/评分对齐）

## 你应该从这个项目学到什么

1. 图像数据如何进入分类流水线（Dataset/Transform/DataLoader）
2. CNN 如何把图像映射为类别 logits
3. 训练/验证/过拟合如何在指标与曲线上体现
4. 如何解读混淆矩阵与误分类样例
5. 如何用可复现的方式做实验与调参，而不是随意改代码

## 快速开始（MNIST）

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行测试：

```bash
pytest -q
```

训练：

```bash
python -m src.train --dataset-name mnist --project-root . --epochs 3 --batch-size 64
```

评估（生成混淆矩阵等图像产物）：

```bash
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pt --dataset-name mnist --project-root .
```

对无标签图片预测并导出：

```bash
python -m src.predict --checkpoint outputs/checkpoints/best_model.pt --image-dir <无标签图片目录> --project-root .
```
