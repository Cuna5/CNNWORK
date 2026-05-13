"""
模型结构分析 (基础任务 3.1.1)
- 逐层输出参数量与特征图尺寸
- 对比 AlexNet 与 VGG16 的参数规模差异
- 输出 Markdown 与 CSV 两种可直接插入报告的表格

运行:
    python analyze.py --dataset cifar100 --input-size 32
"""
import argparse
import csv
import os

from config import Config
from models import build_model
from utils.metrics import count_parameters, layer_summary, format_summary


def _records_to_markdown(records, total_params):
    lines = [
        "| Layer | Type | In Shape | Out Shape | Params |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['name']} | {r['type']} | {r['in_shape']} | {r['out_shape']} | {r['params']:,} |"
        )
    lines.append(f"| **Total** |  |  |  | **{total_params:,} ({total_params/1e6:.2f} M)** |")
    return "\n".join(lines)


def _records_to_csv(records, total_params, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Layer", "Type", "InShape", "OutShape", "Params"])
        for r in records:
            w.writerow([r["name"], r["type"], r["in_shape"], r["out_shape"], r["params"]])
        w.writerow(["Total", "", "", "", total_params])


def analyze_one(name: str, in_channels: int, input_size: int, num_classes: int, out_dir: str, **kwargs):
    model = build_model(name, num_classes=num_classes, in_channels=in_channels, **kwargs)
    info = count_parameters(model)
    records = layer_summary(model, (in_channels, input_size, input_size), device="cpu")
    total = sum(r["params"] for r in records)

    print(f"\n===== {name.upper()}  input=({in_channels},{input_size},{input_size}) =====")
    print(format_summary(records))
    print(f"Total params (all): {info['total_M']:.2f} M")

    # 写文件
    md = _records_to_markdown(records, total)
    with open(os.path.join(out_dir, f"{name}_structure.md"), "w", encoding="utf-8") as f:
        f.write(f"## {name.upper()} on input ({in_channels},{input_size},{input_size})\n\n")
        f.write(md)
        f.write(f"\n\n**All params: {info['total_M']:.2f} M**\n")
    _records_to_csv(records, total, os.path.join(out_dir, f"{name}_structure.csv"))

    return info["total_M"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100",
                    choices=["cifar100", "stl10", "fashion_mnist"])
    ap.add_argument("--input-size", type=int, default=None)
    args = ap.parse_args()

    Config.ensure_dirs()
    out_dir = os.path.join(Config.OUT_DIR, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    orig = {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[args.dataset]
    input_size = args.input_size or orig
    in_channels = 1 if args.dataset == "fashion_mnist" else 3
    num_classes = {"cifar100": 100, "stl10": 10, "fashion_mnist": 10}[args.dataset]

    # AlexNet
    alex_m = analyze_one("alexnet", in_channels, input_size, num_classes, out_dir)

    # VGG16 (默认 fc_dim=1024, 同时额外跑一个 GAP 版本)
    vgg_m = analyze_one("vgg16", in_channels, input_size, num_classes, out_dir,
                        fc_dim=1024, use_gap=False)
    vgg_gap_m = analyze_one("vgg16", in_channels, input_size, num_classes, out_dir,
                            fc_dim=1024, use_gap=True)

    # 对比报告
    report = [
        f"# Model Comparison on {args.dataset} (input={input_size})",
        "",
        "| Model | Total Params (M) |",
        "|---|---|",
        f"| AlexNet | {alex_m:.2f} |",
        f"| VGG16 (fc=1024) | {vgg_m:.2f} |",
        f"| VGG16 (GAP) | {vgg_gap_m:.2f} |",
        "",
        "## Why is VGG16 deeper and larger?",
        "- VGG16 堆叠了 13 个 3x3 卷积层 + 3 个 FC 层, 整体层数显著多于 AlexNet 的 5+3.",
        "- VGG 用连续 3x3 卷积替代 AlexNet 的 5x5/11x11, 感受野等价但参数更可控, "
        "  但由于堆叠更深、通道数增长到 512, 卷积层总参数仍明显更多.",
        "- VGG 的 FC 层若保留原版 4096 维, 单层参数就可达 1 亿以上; "
        "  本实现将其降至 1024 或改用 GAP, 以匹配小分辨率数据集.",
    ]
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    print(f"\n[Done] analysis saved to {out_dir}")


if __name__ == "__main__":
    main()
