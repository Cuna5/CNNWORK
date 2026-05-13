"""
训练与评估循环
"""
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import AverageMeter, accuracy_topk


def train_one_epoch(model, loader, criterion, optimizer, device, topk=(1, 5)):
    model.train()
    loss_m, top1_m, topk_m = AverageMeter(), AverageMeter(), AverageMeter()

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # 若类别数 < topk 的最大值, 跳过 top-k 计算避免报错
        max_k = max(topk)
        if logits.size(1) >= max_k:
            accs = accuracy_topk(logits, y, topk=topk)
            top1_m.update(accs[0], x.size(0))
            topk_m.update(accs[-1], x.size(0))
        else:
            accs = accuracy_topk(logits, y, topk=(1,))
            top1_m.update(accs[0], x.size(0))

        loss_m.update(loss.item(), x.size(0))
        pbar.set_postfix(loss=f"{loss_m.avg:.3f}", top1=f"{top1_m.avg:.2f}")
    return {"loss": loss_m.avg, "top1": top1_m.avg, "topk": topk_m.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device, topk=(1, 5)):
    model.eval()
    loss_m, top1_m, topk_m = AverageMeter(), AverageMeter(), AverageMeter()

    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        max_k = max(topk)
        if logits.size(1) >= max_k:
            accs = accuracy_topk(logits, y, topk=topk)
            top1_m.update(accs[0], x.size(0))
            topk_m.update(accs[-1], x.size(0))
        else:
            accs = accuracy_topk(logits, y, topk=(1,))
            top1_m.update(accs[0], x.size(0))

        loss_m.update(loss.item(), x.size(0))
    return {"loss": loss_m.avg, "top1": top1_m.avg, "topk": topk_m.avg}


def build_optimizer(model, name: str, lr: float, momentum: float, weight_decay: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=True,
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, name: str, epochs: int, step_size: int = 10, gamma: float = 0.1):
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    raise ValueError(f"Unknown scheduler: {name}")


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    optimizer,
    scheduler,
    criterion,
    device,
    ckpt_path: str = None,
) -> Dict[str, List[float]]:
    """完整训练循环, 返回每个 epoch 的指标历史."""
    history = {"train_loss": [], "val_loss": [],
               "train_top1": [], "val_top1": [],
               "train_topk": [], "val_topk": [],
               "lr": []}
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, test_loader, criterion, device)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val["loss"])
        else:
            scheduler.step()

        dt = time.time() - t0
        print(
            f"[Epoch {ep:03d}/{epochs}] lr={lr_now:.4f} "
            f"train_loss={tr['loss']:.4f} train_top1={tr['top1']:.2f} "
            f"val_loss={val['loss']:.4f} val_top1={val['top1']:.2f} val_top5={val['topk']:.2f} "
            f"({dt:.1f}s)"
        )

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(val["loss"])
        history["train_top1"].append(tr["top1"])
        history["val_top1"].append(val["top1"])
        history["train_topk"].append(tr["topk"])
        history["val_topk"].append(val["topk"])
        history["lr"].append(lr_now)

        # 保存最优权重
        if ckpt_path and val["top1"] > best_acc:
            best_acc = val["top1"]
            torch.save({"model": model.state_dict(), "epoch": ep, "acc": best_acc}, ckpt_path)

    return history
