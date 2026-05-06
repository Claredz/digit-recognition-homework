# 手写数字识别作业（MNIST + CNN）

这是一个面向 AI 导论课程的手写数字识别项目。项目目标不是只给出一个准确率数字，而是完整展示从数据理解、预处理、模型训练、评估分析到预测导出的全过程。

当前版本以 MNIST 为主要训练数据，使用一个小型 CNN 作为基线模型，并保留了后续接入老师开放的验证集、测试集或无标签图片的入口。

## 项目目标

本项目希望完成三件事：

1. 训练一个可以识别 0-9 手写数字的 CNN 模型。
2. 在 notebook 中清楚展示技术流程，包括数据分析、模型结构、训练曲线、混淆矩阵和误分类样例。
3. 为后续开放测试集或最终隐藏测试集保留评估与预测导出的接口。

## 推荐使用方式

如果你是第一次学习或准备提交作业，建议优先使用 notebook，而不是直接运行命令行。

推荐顺序：

1. 打开 `submission_notebook.ipynb`
2. 从上到下逐个 cell 运行
3. 观察数据样例、训练曲线、混淆矩阵、分类报告和误分类样例
4. 根据老师最终要求，导出 notebook、实验图片或预测 CSV

说明：

- `submission_notebook.ipynb` 是提交版 notebook，中文正常，已清空输出，适合重新运行后提交。
- `teaching_notebook.ipynb` 是更完整的教学版，内容更详细，适合学习每一步原理。
- `notebook.ipynb` 是工程版入口，复用 `src/` 目录中的模块代码。

## 项目结构

```text
.
├── README.md                         # 项目中文说明
├── requirements.txt                  # Python 依赖列表
├── pytest.ini                        # pytest 测试配置
├── submission_notebook.ipynb          # 推荐提交版 notebook
├── teaching_notebook.ipynb            # 完整教学版 notebook
├── notebook.ipynb                     # 复用 src 模块的工程版 notebook
├── src/
│   ├── config.py                      # 实验配置与输出目录管理
│   ├── data.py                        # MNIST / 文件夹数据集读取与 DataLoader 构建
│   ├── model.py                       # SmallCNN 模型结构
│   ├── engine.py                      # 训练循环、指标保存、曲线绘制
│   ├── train.py                       # 命令行训练入口
│   ├── evaluate.py                    # 命令行评估入口，生成混淆矩阵和误分类图
│   └── predict.py                     # 无标签图片预测与 CSV 导出
├── tests/                             # 自动化测试
├── docs/                              # 设计说明与计划文档
├── data/                              # MNIST 数据下载目录，已被 gitignore 忽略
└── outputs/                           # 训练产物输出目录，已被 gitignore 忽略
```

## 环境准备

建议使用 Python 3.10 或更高版本。当前项目主要依赖 PyTorch、torchvision、matplotlib、scikit-learn、pandas 等库。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如果 notebook 中导入失败，也可以在 notebook 的环境中执行同样的安装命令。

## Notebook 使用说明

### 1. 打开提交版 notebook

推荐打开：

```text
submission_notebook.ipynb
```

这个 notebook 是完整自包含版本，不依赖 `src/` 目录中的代码，适合老师只看一个 notebook 的场景。

运行后会展示：

- MNIST 数据集基本信息
- 类别分布图
- 随机手写数字样例
- 像素分布分析
- 图像预处理过程
- DataLoader batch 形状
- CNN 模型结构和参数量
- logits、softmax、交叉熵损失解释
- 单 batch 反向传播演示
- 完整训练过程
- loss / accuracy 曲线
- 混淆矩阵（带数字标注）
- 每类 precision / recall / F1-score
- 误分类样例分析
- 无标签图片预测和 CSV 导出演示

### 2. 运行顺序

请从上到下顺序运行，不建议跳着运行，因为后面的 cell 会依赖前面定义的变量，例如 `config`、`train_loader`、`model`、`history`、`eval_model`。

如果中途报错，最稳妥的方法是：

1. 重启 notebook kernel
2. 从第一个 cell 重新运行到出错位置
3. 查看是否缺少依赖或数据路径不正确

### 3. 中文显示问题

notebook 中已经配置了 matplotlib 中文字体：

```python
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
```

如果图表标题仍然显示方框，通常是当前系统缺少中文字体，不影响模型训练和结果数据。Windows 环境一般可以正常显示。

## 工程代码使用方式

虽然推荐初学时使用 notebook，但项目也保留了命令行版本，方便后续复现实验或批量预测。

### 训练模型

```bash
python -m src.train --dataset-name mnist --project-root . --epochs 3 --batch-size 64
```

训练完成后主要输出：

```text
outputs/checkpoints/best_model.pt      # 验证集准确率最高的模型
outputs/logs/history.json              # 每轮训练/验证指标
outputs/logs/run_manifest.json         # 本次训练配置摘要
outputs/figures/training_curves.png    # loss 和 accuracy 曲线
```

### 评估模型

```bash
python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pt --dataset-name mnist --project-root .
```

评估完成后主要输出：

```text
outputs/figures/summary.json           # 准确率摘要
outputs/figures/confusion_matrix.csv   # 混淆矩阵原始计数
outputs/figures/confusion_matrix.png   # 带数字标注的混淆矩阵图
outputs/figures/misclassified_grid.png # 误分类样例图
```

注意：当前默认评估使用的是从 MNIST 训练集切分出来的验证集，不是 MNIST 官方 test 集。这样做适合学习和调参；最终成绩仍应以老师提供的测试集或隐藏测试集为准。

### 对无标签图片预测并导出 CSV

```bash
python -m src.predict --checkpoint outputs/checkpoints/best_model.pt --image-dir <无标签图片目录> --project-root .
```

输出文件：

```text
outputs/predictions/predictions.csv
```

CSV 格式：

```csv
filename,prediction
sample_001.png,7
sample_002.png,3
```

## 数据集说明

### MNIST

默认使用 torchvision 自动下载 MNIST：

```text
data/MNIST/
```

MNIST 包含 0-9 共 10 类手写数字图片，图像大小为 28x28，灰度单通道。

### 文件夹格式数据集

如果老师后续开放带标签的验证集，并且目录结构类似下面这样：

```text
external_val/
├── 0/
│   ├── img001.png
│   └── img002.png
├── 1/
│   └── img003.png
└── ...
```

可以使用 `dataset_name='folder'` 读取：

```bash
python -m src.train --dataset-name folder --data-dir external_val --project-root .
```

如果是无标签测试集，不应该用训练入口读取，而应该用 `src.predict` 导出预测 CSV。

## 当前 CNN 架构

项目中的基线模型是 `SmallCNN`：

```text
输入: 1 x 28 x 28

Conv2d(1 -> 32, kernel=3, padding=1)
ReLU
MaxPool2d(2)

Conv2d(32 -> 64, kernel=3, padding=1)
ReLU
MaxPool2d(2)

Flatten
Linear(64 * 7 * 7 -> 128)
ReLU
Dropout(0.25)
Linear(128 -> 10)

输出: 10 个 logits，对应数字 0-9
```

模型参数量约 42 万。对于 MNIST，这个模型已经可以达到较高准确率，适合作为课程作业的可解释基线。

## 如何理解评估结果

### 准确率

准确率表示预测正确的样本占总样本的比例。MNIST 上 98% 左右通常说明基线模型已经工作正常。

### 混淆矩阵

混淆矩阵的含义：

- 行：真实类别
- 列：预测类别
- 对角线：预测正确
- 非对角线：误分类

例如第 4 行第 9 列数字较大，表示真实为 4 的样本经常被预测成 9。

### 误分类样例

误分类不一定都说明模型差。MNIST 中有些样本本身非常潦草，人也难以判断。报告中可以挑几张典型错例，说明：

- 真实标签是什么
- 模型预测成什么
- 模型置信度是多少
- 为什么这个样本容易混淆

这比只写准确率更能体现你理解了模型行为。

## 后续调参方向

当前版本先强调教学和可解释性。等老师开放测试集后，可以再做更高准确率版本。可尝试方向：

1. 增加训练轮数，例如从 3 轮增加到 10 轮。
2. 在训练集上加入轻微数据增强，例如随机旋转和平移。
3. 在卷积层后加入 Batch Normalization。
4. 增加卷积层数量或通道数。
5. 使用学习率调度器，例如验证集 loss 不再下降时降低学习率。
6. 做多组超参数对比，记录每次实验的配置和结果。

注意：不要反复用最终测试集调参，否则会造成测试集信息泄露。调参应该主要依靠训练集切分出的验证集。

## 自动化测试

项目包含基础测试，确保数据读取、模型输出、训练循环、评估和预测导出逻辑正常。

运行：

```bash
pytest -q
```

如果修改了 `src/` 目录中的代码，建议运行测试确认没有破坏原有功能。

## 输出文件与提交建议

默认生成的 `data/` 和 `outputs/` 目录体积较大，已经在 `.gitignore` 中忽略，不建议提交到 git。

课程作业一般建议提交：

```text
submission_notebook.ipynb
README.md
必要的数据报告或截图
预测 CSV（如果老师要求）
```

如果老师只要求 notebook 和数据报告，优先保证 `submission_notebook.ipynb` 能从头到尾运行，并且图表中文显示正常。

## 常见问题

### 1. 为什么混淆矩阵样本总数是 12000？

因为当前默认把 MNIST 训练集 60000 张中的 20% 切分为验证集：

```text
60000 * 0.2 = 12000
```

所以这是正常的，不是官方 10000 张测试集。

### 2. 为什么有些误分类样本我也看不出来？

这是正常现象。MNIST 里确实有一些潦草或边界模糊的样本。可以在报告里把这些样本作为误差分析，而不是简单认为模型失败。

### 3. 提交版 notebook 中文乱码怎么办？

当前 `submission_notebook.ipynb` 已重新以 UTF-8 编码生成。如果再次出现乱码，通常是文件被错误编码方式保存过，建议从 `teaching_notebook.ipynb` 重新生成提交版，而不是手动替换问号。

### 4. 最后老师给无标签测试集怎么办？

使用预测入口：

```bash
python -m src.predict --checkpoint outputs/checkpoints/best_model.pt --image-dir <测试图片目录> --project-root .
```

然后查看：

```text
outputs/predictions/predictions.csv
```

根据老师要求调整 CSV 文件名或列名即可。
