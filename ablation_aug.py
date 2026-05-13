"""
进阶探究 #2: 数据增强策略对比
对比 "basic (仅水平翻转)" 与 "strong (随机裁剪+色彩抖动+Cutout)" 两种策略.

运行:
    python ablation_aug.py --dataset cifar100 --model alexnet --epochs 30
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data import get_dataloaders
from models import build_model
from utils.training import fit, build_optimizer, build_scheduler, evaluate


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_one(dataset, model_name, aug, epochs, args, in_channels, num_classes, input_size):
    set_seed(Config.SEED)
    train_loader, test_loader = get_dataloaders(
        dataset, Config.DATA_ROOT, args.batch_size,
        num_workers=Config.NUM_WORKERS,
        input_size=input_size, aug_strategy=aug,
    )
    if model_name == "alexnet":
        model = build_model("alexnet", num_classes=num_classes, in_channels=in_channels)
    else:
        model = build_model("vgg16", num_classes=num_classes, in_channels=in_channels,
                            fc_dim=1024, use_gap=False)
    model = model.to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, "sgd", args.lr, Config.MOMENTUM, Config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, "cosine", epochs)

    history = fit(model, train_loader, test_loader, epochs,
                  optimizer, scheduler, criterion, Config.DEVICE,
                  ckpt_path=None)
    final = evaluate(model, test_loader, criterion, Config.DEVICE)
    return history, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--model", default="alexnet", choices=["alexnet", "vgg16"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    Config.ensure_dirs()
    out_dir = os.path.join(Config.OUT_DIR, f"ablation_aug_{args.dataset}_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    orig = {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[args.dataset]
    in_channels = 1 if args.dataset == "fashion_mnist" else 3
    num_classes = {"cifar100": 100, "stl10": 10, "fashion_mnist": 10}[args.dataset]

    results = {}
    for aug in ["basic", "strong"]:
        print(f"\n========== Training with aug={aug} ==========")
        hist, final = run_one(args.dataset, args.model, aug, args.epochs,
                              args, in_channels, num_classes, orig)
        results[aug] = {"history": hist, "final": final}

    # 保存对比曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for aug, color in zip(["basic", "strong"], ["tab:blue", "tab:orange"]):
        h = results[aug]["history"]
        eps = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(eps, h["val_loss"], label=f"{aug}", color=color)
        axes[1].plot(eps, h["val_top1"], label=f"{aug}", color=color)
    axes[0].set_title("Val Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Val Top-1 (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aug_compare.png"), dpi=150)
    plt.close(fig)

    # 保存数值结果与简短分析
    summary = {
        aug: {"final_top1": r["final"]["top1"], "final_top5": r["final"]["topk"]}
        for aug, r in results.items()
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary,
                   "history": {k: v["history"] for k, v in results.items()}},
                  f, indent=2)

    # 对比分析文本
    diff = summary["strong"]["final_top1"] - summary["basic"]["final_top1"]
    note = (
        f"# Data Augmentation Ablation ({args.dataset}, {args.model})\n\n"
        f"| Strategy | Top-1 (%) | Top-5 (%) |\n|---|---|---|\n"
        f"| basic (Flip)               | {summary['basic']['final_top1']:.2f} | {summary['basic']['final_top5']:.2f} |\n"
        f"| strong (Crop+Jitter+Cutout)| {summary['strong']['final_top1']:.2f} | {summary['strong']['final_top5']:.2f} |\n\n"
        f"组合增强相对基础增强的 Top-1 提升: **{diff:+.2f}%**.\n"
        "组合增强通过增加训练分布多样性 (平移、颜色、遮挡), 减少过拟合, "
        "一般能显著提升测试集泛化性能, 在类间差异小的 CIFAR-100 上收益更明显.\n"
    )
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(note)
    print(note)
    print(f"[Done] ablation saved to {out_dir}")


if __name__ == "__main__":
    main()
