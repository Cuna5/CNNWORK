"""
进阶探究 #4: 网络特征可视化
- 可视化网络前几层卷积核 (学到的边缘/纹理基础特征)
- 选取单张测试图片, 可视化网络中间层特征图

运行:
    python visualize_features.py --dataset cifar100 --model alexnet \
        --ckpt outputs/cifar100_alexnet_run/best.pt
"""
import argparse
import os
import torch

from config import Config
from data import get_dataloaders
from models import build_model
from utils.visualization import visualize_filters, visualize_feature_maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--model", default="alexnet", choices=["alexnet", "vgg16"])
    ap.add_argument("--ckpt", required=True, help="训练好的权重 .pt 路径")
    ap.add_argument("--n-layers", type=int, default=4, help="可视化前多少个卷积层的特征图")
    args = ap.parse_args()

    Config.ensure_dirs()
    out_dir = os.path.join(Config.OUT_DIR, f"features_{args.dataset}_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    orig = {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[args.dataset]
    in_channels = 1 if args.dataset == "fashion_mnist" else 3
    num_classes = {"cifar100": 100, "stl10": 10, "fashion_mnist": 10}[args.dataset]

    # 构建模型并载入权重
    if args.model == "alexnet":
        model = build_model("alexnet", num_classes=num_classes, in_channels=in_channels)
    else:
        model = build_model("vgg16", num_classes=num_classes, in_channels=in_channels,
                            fc_dim=1024, use_gap=False)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    print(f"[Load] ckpt epoch={state.get('epoch','?')} acc={state.get('acc','?')}")

    # 取一张测试样本
    _, test_loader = get_dataloaders(args.dataset, Config.DATA_ROOT,
                                     batch_size=8, num_workers=0,
                                     input_size=orig, aug_strategy="basic")
    x, _ = next(iter(test_loader))
    sample = x[0]

    # 前两层卷积核
    for li in range(2):
        visualize_filters(model, os.path.join(out_dir, f"filters_layer{li}.png"),
                          layer_idx=li, max_filters=64)

    # 前 N 个卷积层的特征图
    visualize_feature_maps(model, sample, os.path.join(out_dir, "feature_maps.png"),
                           max_layers=args.n_layers, max_maps=16, device="cpu")
    print(f"[Done] saved to {out_dir}")


if __name__ == "__main__":
    main()
