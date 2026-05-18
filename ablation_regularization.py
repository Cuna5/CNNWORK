"""
进阶探究 #4: 正则化策略对比
对比三种正则化策略对模型收敛效果与测试精度的影响:
  - 不同 Dropout 比例 (0 / 0.3 / 0.5)
  - Batch Normalization 开关 (仅 VGG16 有效)
  - L2 权重衰减 (0 / 1e-4 / 5e-4)

运行:
    python ablation_regularization.py --dataset cifar100 --model alexnet --epochs 30
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


def run_one(dataset, model_name, epochs, args, in_channels, num_classes,
            dropout, weight_decay, batch_norm):
    set_seed(Config.SEED)
    orig = {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[dataset]
    train_loader, test_loader = get_dataloaders(
        dataset, Config.DATA_ROOT, args.batch_size,
        num_workers=Config.NUM_WORKERS,
        input_size=orig, aug_strategy="basic",
    )
    if model_name == "alexnet":
        model = build_model("alexnet", num_classes=num_classes,
                            in_channels=in_channels, dropout=dropout)
    else:
        model = build_model("vgg16", num_classes=num_classes,
                            in_channels=in_channels, fc_dim=1024,
                            use_gap=False, dropout=dropout, batch_norm=batch_norm)
    model = model.to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, "sgd", args.lr, Config.MOMENTUM, weight_decay)
    scheduler = build_scheduler(optimizer, "cosine", epochs)

    history = fit(model, train_loader, test_loader, epochs,
                  optimizer, scheduler, criterion, Config.DEVICE, ckpt_path=None)
    final = evaluate(model, test_loader, criterion, Config.DEVICE)
    return history, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=["cifar100", "stl10", "fashion_mnist"])
    ap.add_argument("--model", default="alexnet", choices=["alexnet", "vgg16"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    Config.ensure_dirs()
    out_dir = os.path.join(Config.OUT_DIR, f"ablation_reg_{args.dataset}_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    in_channels = 1 if args.dataset == "fashion_mnist" else 3
    num_classes = {"cifar100": 100, "stl10": 10, "fashion_mnist": 10}[args.dataset]

    # (label, dropout, weight_decay, batch_norm)
    experiments = [
        ("dropout=0.0", 0.0, Config.WEIGHT_DECAY, True),
        ("dropout=0.3", 0.3, Config.WEIGHT_DECAY, True),
        ("dropout=0.5", 0.5, Config.WEIGHT_DECAY, True),
        ("wd=0",        0.5, 0.0,  True),
        ("wd=1e-4",     0.5, 1e-4, True),
    ]
    if args.model == "vgg16":
        experiments.append(("no_bn", 0.5, Config.WEIGHT_DECAY, False))

    results = {}
    for label, dropout, wd, bn in experiments:
        print(f"\n========== {label} ==========")
        hist, final = run_one(args.dataset, args.model, args.epochs, args,
                              in_channels, num_classes, dropout, wd, bn)
        results[label] = {"history": hist, "final": final,
                          "dropout": dropout, "weight_decay": wd, "batch_norm": bn}

    groups = {
        "Dropout": [k for k in results if k.startswith("dropout")],
        "L2_Weight_Decay": [k for k in results if k.startswith("wd")],
    }
    if args.model == "vgg16":
        groups["Batch_Normalization"] = ["dropout=0.5", "no_bn"]

    for group_name, keys in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for key, color in zip(keys, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
            h = results[key]["history"]
            eps = range(1, len(h["val_top1"]) + 1)
            axes[0].plot(eps, h["val_loss"], label=key, color=color)
            axes[1].plot(eps, h["val_top1"], label=key, color=color)
        axes[0].set_title(f"{group_name} - Val Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set_title(f"{group_name} - Val Top-1 (%)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"reg_{group_name.lower()}.png"), dpi=150)
        plt.close(fig)

    summary = {
        k: {"final_top1": v["final"]["top1"], "final_top5": v["final"]["topk"],
            "dropout": v["dropout"], "weight_decay": v["weight_decay"],
            "batch_norm": v["batch_norm"]}
        for k, v in results.items()
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"summary": summary,
                   "history": {k: v["history"] for k, v in results.items()}}, f, indent=2)

    best_key = max(summary, key=lambda k: summary[k]["final_top1"])
    rows = "\n".join(
        f"| {k} | {v['dropout']} | {v['weight_decay']} | {v['batch_norm']} "
        f"| {v['final_top1']:.2f} | {v['final_top5']:.2f} |"
        for k, v in summary.items()
    )
    bn_line = "- **Batch Normalization**: BN 对每层输入做归一化，加速收敛并起到隐式正则化作用，通常能显著提升精度（仅 VGG16 可对比）。\n" if args.model == "vgg16" else ""
    report = (
        f"# 正则化策略对比 ({args.dataset}, {args.model})\n\n"
        f"| 实验 | Dropout | Weight Decay | BN | Top-1 (%) | Top-5 (%) |\n"
        f"|---|---|---|---|---|---|\n{rows}\n\n"
        f"最优配置: **{best_key}**，Top-1={summary[best_key]['final_top1']:.2f}%.\n\n"
        "分析:\n"
        "- **Dropout**: 适度 Dropout（0.3~0.5）通过随机失活神经元抑制过拟合，过高则欠拟合，过低则泛化差。\n"
        "- **L2 权重衰减**: 惩罚大权重，防止模型过度依赖少数特征；过大的衰减系数会限制模型容量。\n"
        f"{bn_line}"
        "三种策略可叠加使用，最优方案通常为适度 Dropout + 合理 L2 + BN 开启的组合。\n"
    )
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"[Done] outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
