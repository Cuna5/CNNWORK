"""
可视化工具: 训练曲线 / 混淆矩阵 / 样本预测 / 卷积核与中间特征图
"""
import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能保存图像
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_curves(history: dict, out_path: str, topk_label: str = "Top-5"):
    """绘制训练/验证 loss 与 top1/top-k 曲线."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_top1"], label="train top1")
    axes[1].plot(epochs, history["val_top1"], label="val top1")
    if any(v > 0 for v in history.get("val_topk", [])):
        axes[1].plot(epochs, history["val_topk"], label=f"val {topk_label}", linestyle="--")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("acc")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_confusion_matrix(model, loader, class_names, device, out_path: str, top_pairs: int = 5):
    """绘制混淆矩阵, 并打印 top_pairs 个最易混淆的类别对."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    # 类别多时不显示具体文字, 仅显示刻度
    if len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # 找出高频易混淆类别对 (off-diagonal 最大的几项)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    pairs = []
    flat = np.argsort(cm_off.flatten())[::-1]
    for idx in flat[:top_pairs]:
        i, j = divmod(idx, cm.shape[1])
        if cm_off[i, j] == 0:
            break
        pairs.append((class_names[i], class_names[j], int(cm_off[i, j])))
    return pairs


@torch.no_grad()
def visualize_predictions(model, loader, class_names, device, out_path: str,
                          n_correct: int = 6, n_wrong: int = 6, mean=None, std=None):
    """随机抽样, 可视化若干正确与错误预测样本."""
    model.eval()
    correct_imgs, wrong_imgs = [], []
    for x, y in loader:
        x_dev = x.to(device)
        logits = model(x_dev)
        pred = logits.argmax(1).cpu()
        for i in range(x.size(0)):
            info = (x[i], int(y[i]), int(pred[i]))
            if pred[i] == y[i] and len(correct_imgs) < n_correct:
                correct_imgs.append(info)
            elif pred[i] != y[i] and len(wrong_imgs) < n_wrong:
                wrong_imgs.append(info)
            if len(correct_imgs) >= n_correct and len(wrong_imgs) >= n_wrong:
                break
        if len(correct_imgs) >= n_correct and len(wrong_imgs) >= n_wrong:
            break

    samples = correct_imgs + wrong_imgs
    n = len(samples)
    cols = 6
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)

    for idx, (img, true_y, pred_y) in enumerate(samples):
        ax = axes[idx // cols][idx % cols]
        img_np = img.cpu().numpy()
        if mean is not None and std is not None:
            mean_arr = np.array(mean).reshape(-1, 1, 1)
            std_arr = np.array(std).reshape(-1, 1, 1)
            img_np = img_np * std_arr + mean_arr
        img_np = np.clip(img_np, 0, 1)
        if img_np.shape[0] == 1:
            ax.imshow(img_np[0], cmap="gray")
        else:
            ax.imshow(np.transpose(img_np, (1, 2, 0)))
        ok = (true_y == pred_y)
        color = "green" if ok else "red"
        ax.set_title(f"T:{class_names[true_y][:10]}\nP:{class_names[pred_y][:10]}",
                     color=color, fontsize=8)
        ax.axis("off")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_filters(model: nn.Module, out_path: str, layer_idx: int = 0, max_filters: int = 64):
    """可视化某个卷积层的卷积核权重."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return
    conv = convs[min(layer_idx, len(convs) - 1)]
    w = conv.weight.data.clone().cpu()
    # 归一化到 [0,1]
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)

    n = min(max_filters, w.size(0))
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(n):
        ax = axes[i // cols][i % cols]
        kernel = w[i]
        if kernel.shape[0] == 1:
            ax.imshow(kernel[0], cmap="gray")
        elif kernel.shape[0] == 3:
            ax.imshow(np.transpose(kernel.numpy(), (1, 2, 0)))
        else:
            # 多通道取均值可视化
            ax.imshow(kernel.mean(0).numpy(), cmap="viridis")
        ax.axis("off")
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle(f"Conv layer #{layer_idx} filters")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def visualize_feature_maps(model: nn.Module, sample: torch.Tensor, out_path: str,
                           max_layers: int = 4, max_maps: int = 16, device="cpu"):
    """可视化输入一张图像后, 前 max_layers 个卷积层输出的特征图."""
    model = model.to(device).eval()
    sample = sample.to(device)
    if sample.dim() == 3:
        sample = sample.unsqueeze(0)

    feats: List[torch.Tensor] = []
    hooks = []

    def _hook(module, inp, out):
        feats.append(out.detach().cpu())

    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    for m in convs[:max_layers]:
        hooks.append(m.register_forward_hook(_hook))
    model(sample)
    for h in hooks:
        h.remove()

    fig, axes = plt.subplots(len(feats), max_maps, figsize=(max_maps, len(feats) * 1.2))
    if len(feats) == 1:
        axes = axes.reshape(1, -1)
    for li, fm in enumerate(feats):
        fm0 = fm[0]  # 第一个样本
        for j in range(max_maps):
            ax = axes[li][j]
            if j < fm0.size(0):
                m = fm0[j].numpy()
                m = (m - m.min()) / (m.max() - m.min() + 1e-8)
                ax.imshow(m, cmap="viridis")
            ax.axis("off")
        axes[li][0].set_ylabel(f"L{li}", rotation=0, labelpad=20)
    fig.suptitle("Feature maps (first N conv layers)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
