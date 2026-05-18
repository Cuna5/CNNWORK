"""
进阶探究 #3: 输入分辨率对比实验
对比数据集原始分辨率与缩放至 224×224 两种设置下的 Top-1 精度与每 epoch 训练耗时.

运行:
    python ablation_resolution.py --dataset cifar100 --model alexnet --epochs 30
"""
import argparse
import json
import os
import random
import time

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


def run_one(dataset, model_name, input_size, epochs, args, in_channels, num_classes):
    set_seed(Config.SEED)
    train_loader, test_loader = get_dataloaders(
        dataset, Config.DATA_ROOT, args.batch_size,
        num_workers=Config.NUM_WORKERS,
        input_size=input_size, aug_strategy="basic",
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

    # 记录每 epoch 耗时
    epoch_times = []
    history = {"train_loss": [], "val_loss": [], "train_top1": [], "val_top1": [],
               "train_topk": [], "val_topk": [], "lr": []}
    best_acc = 0.0

    from utils.training import train_one_epoch
    for ep in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val = evaluate(model, test_loader, criterion, Config.DEVICE)
        scheduler.step()
        dt = time.time() - t0
        epoch_times.append(dt)
        print(f"[res={input_size} ep={ep:03d}] top1={val['top1']:.2f}% ({dt:.1f}s)")
        history["train_loss"].append(tr["loss"]); history["val_loss"].append(val["loss"])
        history["train_top1"].append(tr["top1"]); history["val_top1"].append(val["top1"])
        history["train_topk"].append(tr["topk"]); history["val_topk"].append(val["topk"])
        history["lr"].append(lr_now)

    final = evaluate(model, test_loader, criterion, Config.DEVICE)
    return history, final, epoch_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=["cifar100", "stl10", "fashion_mnist"])
    ap.add_argument("--model", default="alexnet", choices=["alexnet", "vgg16"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    Config.ensure_dirs()
    out_dir = os.path.join(Config.OUT_DIR, f"ablation_resolution_{args.dataset}_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    orig = {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[args.dataset]
    in_channels = 1 if args.dataset == "fashion_mnist" else 3
    num_classes = {"cifar100": 100, "stl10": 10, "fashion_mnist": 10}[args.dataset]

    results = {}
    for res in [orig, 224]:
        label = f"{res}x{res}"
        print(f"\n========== input_size={res} ==========")
        hist, final, times = run_one(args.dataset, args.model, res, args.epochs,
                                     args, in_channels, num_classes)
        results[label] = {"history": hist, "final": final,
                          "avg_epoch_time": float(np.mean(times))}

    # 对比曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["tab:blue", "tab:orange"]
    labels = list(results.keys())
    for label, color in zip(labels, colors):
        h = results[label]["history"]
        eps = range(1, len(h["val_top1"]) + 1)
        axes[0].plot(eps, h["val_loss"], label=label, color=color)
        axes[1].plot(eps, h["val_top1"], label=label, color=color)
    axes[0].set_title("Val Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Val Top-1 (%)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "resolution_compare.png"), dpi=150)
    plt.close(fig)

    summary = {
        label: {
            "final_top1": r["final"]["top1"],
            "final_top5": r["final"]["topk"],
            "avg_epoch_time_s": r["avg_epoch_time"],
        }
        for label, r in results.items()
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary,
                   "history": {k: v["history"] for k, v in results.items()}}, f, indent=2)

    rows = "\n".join(
        f"| {lbl} | {v['final_top1']:.2f} | {v['final_top5']:.2f} | {v['avg_epoch_time_s']:.1f} |"
        for lbl, v in summary.items()
    )
    diff_top1 = summary[labels[1]]["final_top1"] - summary[labels[0]]["final_top1"]
    speed_ratio = summary[labels[1]]["avg_epoch_time_s"] / summary[labels[0]]["avg_epoch_time_s"]
    report = (
        f"# 输入分辨率对比实验 ({args.dataset}, {args.model})\n\n"
        f"| 分辨率 | Top-1 (%) | Top-5 (%) | 平均 epoch 耗时 (s) |\n|---|---|---|---|\n"
        f"{rows}\n\n"
        f"224×224 相对原始分辨率 Top-1 变化: **{diff_top1:+.2f}%**，"
        f"训练速度比: **{speed_ratio:.2f}x**（>1 表示 224 更慢）.\n\n"
        "分析: 高分辨率输入保留更多空间细节，有助于提升精度，但特征图更大导致计算量显著增加，"
        "训练速度下降明显。对于本身分辨率较低的数据集（如 CIFAR-100 32×32），"
        "上采样至 224 引入的插值伪影可能抵消分辨率优势，精度提升有限甚至下降。\n"
    )
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"[Done] outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
