"""
学生提交模板：predict.py
========================================
系统会以如下命令运行您的脚本：
  python3 predict.py --testdata /testdata --output /results/submission.csv
"""

import argparse
import os
import struct
import numpy as np
from pathlib import Path

def load_mnist_images(filepath):
    """
    读取 MNIST IDX3-UBYTE 图像文件。
    返回: (N, 784) float32 归一化到 [0, 1]
    """
    with open(filepath, 'rb') as f:
        # 读取文件头: magic number, num_images, rows, cols
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in images file")
        
        # 读取像素数据
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(n, rows * cols).astype(np.float32) / 255.0
        return images

def predict(testdata_dir: str, output_path: str):
    # 1. 加载测试图像 (IDX 格式)
    # 约定文件名为 test_B_images.idx3-ubyte
    test_images_path = Path(testdata_dir) / "test_B_images.idx3-ubyte"
    if not test_images_path.exists():
        print(f"错误：找不到测试图像文件 {test_images_path}")
        return

    try:
        images = load_mnist_images(test_images_path)
        print(f"成功加载测试集: {images.shape}")
    except Exception as e:
        print(f"加载测试集失败: {e}")
        return

    # 2. ── 在此编写您的模型推理代码 ─────────────────────────
    # 示例：加载模型并进行预测
    # import torch
    # model = MyModel()
    # model.load_state_dict(torch.load("model.pth"))
    # preds = model(torch.from_numpy(images)).argmax(dim=1).numpy()
    
    # 这里我们使用全 0 作为占位符，请替换为您的真实预测
    preds = np.zeros(len(images), dtype=int)
    # ──────────────────────────────────────────────────────

    # 3. 写入结果 submission.csv (无表头，每行一个数字)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(f"{p}\n")

    print(f"预测完成，结果已写入：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testdata", required=True, help="测试数据目录")
    parser.add_argument("--output",   required=True, help="预测结果输出路径")
    args = parser.parse_args()
    predict(args.testdata, args.output)
