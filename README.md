# 基于经典 CNN 的图像分类实验

从零实现 **AlexNet** 与 **VGG16**，在 **CIFAR-100 / STL-10 / Fashion-MNIST** 数据集上完成图像分类全流程（数据加载 → 预处理 → 模型搭建 → 训练 → 评估 → 可视化 → 进阶对比实验）。

## 目录结构

```
cvleaf/
├── README.md
├── requirements.txt
├── config.py                  全局超参数配置
├── main.py                    训练主入口
├── analyze.py                 参数量与特征图尺寸分析 (基础任务 3.1.1)
├── ablation_aug.py            进阶 #2: 数据增强策略对比
├── visualize_features.py      进阶 #4: 卷积核 & 特征图可视化
├── models/
│   ├── alexnet.py             AlexNet (小分辨率适配)
│   └── vgg16.py               VGG16 (BN + 显存优化)
├── data/
│   └── datasets.py            数据集加载 + 数据增强 (含 Cutout)
└── utils/
    ├── metrics.py             参数统计 / 逐层 summary / Top-k 准确率
    ├── training.py            训练 & 评估循环
    └── visualization.py       曲线 / 混淆矩阵 / 预测样例 / 特征可视化
```

## 环境

```bash
pip install -r requirements.txt
```

建议 GPU。代码自动检测 `cuda`，无 GPU 也能在小数据集（Fashion-MNIST）上运行。

## 快速开始

### 1. 模型结构分析（参数量 + 逐层特征图尺寸表）

```bash
python analyze.py --dataset cifar100
```

输出到 `outputs/analysis/`：
- `alexnet_structure.md` / `.csv`
- `vgg16_structure.md` / `.csv`（含 GAP 版对比）
- `comparison.md`（AlexNet vs VGG16 参数量差异分析）

### 2. 基础训练 (3.1.2)

```bash
# AlexNet on CIFAR-100
python main.py --dataset cifar100 --model alexnet --epochs 30 --tag base

# VGG16 on CIFAR-100 (显存敏感, 默认 fc_dim=1024)
python main.py --dataset cifar100 --model vgg16 --epochs 30 --tag base

# VGG16 + 全局平均池化 (进一步降参)
python main.py --dataset cifar100 --model vgg16 --vgg-use-gap --tag gap

# Fashion-MNIST (单通道自动适配)
python main.py --dataset fashion_mnist --model alexnet --epochs 30
```

运行后输出到 `outputs/<dataset>_<model>_<tag>/`：
- `layer_summary.txt` 逐层参数 / 输入输出形状
- `history.json` 训练历史
- `curves.png` loss 与 Top-1/Top-5 曲线
- `confusion_matrix.png` + `top_confused_pairs.json` 混淆矩阵与易混类别
- `predictions.png` 正确/错误预测样例
- `best.pt` 最优权重

### 3. 进阶实验

#### (a) 数据增强策略对比（进阶 #2）

```bash
python ablation_aug.py --dataset cifar100 --model alexnet --epochs 30
```

对比 `basic (仅水平翻转)` 与 `strong (RandomCrop + ColorJitter + Cutout)`，输出到
`outputs/ablation_aug_cifar100_alexnet/`，含对比曲线、Top-1 数值与分析报告。

#### (b) 卷积核与中间层特征图可视化（进阶 #4）

```bash
python visualize_features.py --dataset cifar100 --model alexnet \
    --ckpt outputs/cifar100_alexnet_base/best.pt
```

输出 `filters_layer0/1.png`（学到的边缘/纹理滤波器）与 `feature_maps.png`（前 4 层卷积的特征图）。

## 训练配置说明

| 项目 | 默认 | 说明 |
|---|---|---|
| 损失函数 | CrossEntropy | 多分类任务标准选择 |
| 优化器 | SGD(momentum=0.9, nesterov) | 小模型经典选择, 泛化更稳 |
| 学习率调度 | CosineAnnealing | 无需调 step, 尾部退火平滑 |
| 初始学习率 | 0.01 | SGD 典型取值 |
| 权重衰减 | 5e-4 | L2 正则化 |
| Batch Size | 128 | 32x32 图单卡显存友好 |
| Epochs | 30 | 作业最低要求 |

可在命令行覆盖，如 `--optimizer adam --scheduler step`。

## 小分辨率适配要点

- **AlexNet**: 原始 11×11 卷积核对 32×32 过大，改为 3×3 + padding=1，池化减少，`AdaptiveAvgPool2d` 兜底。
- **VGG16**: 默认 BN 开启；全连接 4096→1024；支持 `--vgg-use-gap` 用全局平均池化替代 FC，大幅降参。
- **Fashion-MNIST**: 单通道自动处理（`in_channels=1`）。

## 输出文件一览

每次训练都会生成：
- 模型结构表（层类型、输入/输出形状、参数量）
- 训练曲线图
- 混淆矩阵 + 高频易混淆类别对
- 正确/错误预测样本可视化
- 最终 Top-1 / Top-5 准确率（CIFAR-100 按作业要求额外统计 Top-5）

以上全部可直接粘贴进实验报告。
