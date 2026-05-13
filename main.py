"""
主训练入口
支持通过命令行选择数据集 / 模型 / 增强策略等.

示例:
    python main.py --dataset cifar100 --model alexnet --epochs 30
    python main.py --dataset cifar100 --model vgg16   --epochs 30 --vgg-use-gap
    python main.py --dataset fashion_mnist --model alexnet
"""
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn

from config import Config
from data import get_dataloaders, get_class_names
from data.datasets import _STATS
from models import build_model
from utils.metrics import count_parameters, layer_summary, format_summary
from utils.training import fit, build_optimizer, build_scheduler, evaluate
from utils.visualization import (
    plot_curves,
    plot_confusion_matrix,
    visualize_predictions,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 允许 cudnn 自动寻找最快算法 (略损失可复现性, 换取训练速度)
    torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=Config.DATASET,
                   choices=["cifar100", "stl10", "fashion_mnist"])
    p.add_argument("--model", default=Config.MODEL, choices=["alexnet", "vgg16"])
    p.add_argument("--epochs", type=int, default=Config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=Config.LR)
    p.add_argument("--optimizer", default=Config.OPTIMIZER, choices=["sgd", "adam"])
    p.add_argument("--scheduler", default=Config.SCHEDULER, choices=["cosine", "step", "plateau"])
    p.add_argument("--aug", default=Config.AUG_STRATEGY, choices=["basic", "strong"])
    p.add_argument("--input-size", type=int, default=None,
                   help="输入尺寸, 不指定则用数据集原始尺寸")
    p.add_argument("--vgg-fc-dim", type=int, default=Config.VGG_FC_DIM)
    p.add_argument("--vgg-use-gap", action="store_true", help="VGG 使用全局平均池化")
    p.add_argument("--tag", default="run", help="本次实验标签, 用于区分输出目录")
    p.add_argument("--eval-only", action="store_true", help="仅评估已训练模型")
    return p.parse_args()


def main():
    args = parse_args()
    Config.ensure_dirs()
    set_seed(Config.SEED)

    # 子目录: outputs/<dataset>_<model>_<tag>/
    run_name = f"{args.dataset}_{args.model}_{args.tag}"
    run_dir = os.path.join(Config.OUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 输入尺寸
    orig = {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[args.dataset]
    input_size = args.input_size or orig
    in_channels = 1 if args.dataset == "fashion_mnist" else 3
    num_classes = {"cifar100": 100, "stl10": 10, "fashion_mnist": 10}[args.dataset]

    # ---------- 数据 ----------
    print(f"[Data] dataset={args.dataset} aug={args.aug} input_size={input_size}")
    train_loader, test_loader = get_dataloaders(
        args.dataset, Config.DATA_ROOT,
        batch_size=args.batch_size,
        num_workers=Config.NUM_WORKERS,
        input_size=input_size,
        aug_strategy=args.aug,
    )

    # ---------- 模型 ----------
    if args.model == "alexnet":
        model = build_model("alexnet", num_classes=num_classes,
                            in_channels=in_channels, dropout=Config.DROPOUT)
    else:
        model = build_model("vgg16", num_classes=num_classes,
                            in_channels=in_channels,
                            fc_dim=args.vgg_fc_dim,
                            use_gap=args.vgg_use_gap,
                            dropout=Config.DROPOUT)
    model = model.to(Config.DEVICE)

    # 参数量与层级信息
    info = count_parameters(model)
    print(f"[Model] {args.model}  total={info['total_M']:.2f}M  trainable={info['trainable_M']:.2f}M")
    records = layer_summary(model, (in_channels, input_size, input_size), device="cpu")
    summary_txt = format_summary(records)
    print(summary_txt)
    with open(os.path.join(run_dir, "layer_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary_txt + "\n")
    model = model.to(Config.DEVICE)

    # ---------- 训练 ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.optimizer, args.lr,
                                Config.MOMENTUM, Config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs,
                                Config.STEP_SIZE, Config.GAMMA)
    ckpt = os.path.join(run_dir, "best.pt")

    if not args.eval_only:
        print(f"[Train] optimizer={args.optimizer} sched={args.scheduler} epochs={args.epochs}")
        history = fit(model, train_loader, test_loader,
                      args.epochs, optimizer, scheduler, criterion,
                      Config.DEVICE, ckpt_path=ckpt)
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        plot_curves(history, os.path.join(run_dir, "curves.png"))

    # 加载最优权重做最终评估与可视化
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=Config.DEVICE)
        model.load_state_dict(state["model"])
        print(f"[Load] best checkpoint from epoch {state['epoch']}, acc={state['acc']:.2f}")

    final = evaluate(model, test_loader, criterion, Config.DEVICE)
    print(f"[Final] loss={final['loss']:.4f}  Top-1={final['top1']:.2f}%  Top-5={final['topk']:.2f}%")
    with open(os.path.join(run_dir, "final_metrics.json"), "w") as f:
        json.dump({"test": final, "params_M": info["total_M"]}, f, indent=2)

    # 混淆矩阵 + 预测样例
    class_names = get_class_names(args.dataset)
    top_pairs = plot_confusion_matrix(
        model, test_loader, class_names, Config.DEVICE,
        out_path=os.path.join(run_dir, "confusion_matrix.png"),
    )
    print("[Top confused pairs]", top_pairs)
    with open(os.path.join(run_dir, "top_confused_pairs.json"), "w") as f:
        json.dump(top_pairs, f, indent=2, ensure_ascii=False)

    mean, std = _STATS[args.dataset]
    visualize_predictions(
        model, test_loader, class_names, Config.DEVICE,
        out_path=os.path.join(run_dir, "predictions.png"),
        mean=mean, std=std,
    )
    print(f"[Done] outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
