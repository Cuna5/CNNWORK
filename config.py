"""
全局实验配置
统一管理数据集、模型、训练超参数与路径，便于复现与对比实验。
"""
import os
import torch


class Config:
    # ============ 基础环境 ============
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Windows 下建议 0~4

    # ============ 数据集 ============
    # 可选: "cifar100" | "stl10" | "fashion_mnist"
    DATASET = "cifar100"
    DATA_ROOT = "./data_files"
    # 原始分辨率走小图分支, 进阶实验可切换为 224 做分辨率对比
    INPUT_SIZE = 32  # cifar100=32, stl10=96, fashion_mnist=28

    # ============ 模型 ============
    # 可选: "alexnet" | "vgg16"
    MODEL = "alexnet"
    # VGG16 显存优化: 将 4096 降为 1024, 或用全局平均池化 (GAP)
    VGG_FC_DIM = 1024
    VGG_USE_GAP = False  # True 时用全局平均池化替代传统全连接层
    DROPOUT = 0.5

    # ============ 训练 ============
    BATCH_SIZE = 128
    EPOCHS = 30
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    OPTIMIZER = "sgd"  # "sgd" | "adam"
    SCHEDULER = "cosine"  # "cosine" | "step" | "plateau"
    STEP_SIZE = 10
    GAMMA = 0.1

    # ============ 数据增强 ============
    # "basic": 仅标准化 + 随机水平翻转
    # "strong": 组合增强 (RandomCrop + ColorJitter + Cutout)
    AUG_STRATEGY = "basic"

    # ============ 输出路径 ============
    OUT_DIR = "./outputs"
    CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUT_DIR, "logs")
    FIG_DIR = os.path.join(OUT_DIR, "figures")

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.OUT_DIR, cls.CKPT_DIR, cls.LOG_DIR, cls.FIG_DIR, cls.DATA_ROOT]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def num_classes(cls):
        return {"cifar100": 100, "stl10": 10, "fashion_mnist": 10}[cls.DATASET]

    @classmethod
    def in_channels(cls):
        return 1 if cls.DATASET == "fashion_mnist" else 3

    @classmethod
    def default_size(cls):
        return {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[cls.DATASET]
