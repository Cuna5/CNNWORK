"""
模型结构分析与训练指标
- count_parameters: 统计总参数量 / 可训练参数量 (Million)
- layer_summary:    逐层列出卷积/全连接参数量与输出特征图尺寸
- accuracy_topk:    Top-1 / Top-5 准确率
- AverageMeter:     训练过程中的指标累积
"""
from typing import List, Dict, Tuple
import torch
import torch.nn as nn


class AverageMeter:
    """对 loss/acc 取运行均值的小工具."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """返回模型总参数量与可训练参数量 (单位: M)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }


@torch.no_grad()
def layer_summary(model: nn.Module, input_size: Tuple[int, int, int], device="cpu") -> List[Dict]:
    """逐层统计输出尺寸与参数量, 返回一个可打印/导出表格的 list.

    input_size: (C, H, W)
    """
    model = model.to(device).eval()
    records: List[Dict] = []
    hooks = []

    def _hook(name):
        def _fn(module, inp, out):
            n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            in_shape = tuple(inp[0].shape[1:]) if isinstance(inp, tuple) else tuple(inp.shape[1:])
            out_shape = tuple(out.shape[1:])
            records.append({
                "name": name,
                "type": module.__class__.__name__,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "params": n_params,
            })
        return _fn

    # 仅挂载叶子模块
    for name, m in model.named_modules():
        if len(list(m.children())) == 0 and not isinstance(m, nn.Dropout):
            hooks.append(m.register_forward_hook(_hook(name)))

    dummy = torch.zeros(1, *input_size, device=device)
    model(dummy)
    for h in hooks:
        h.remove()
    return records


def format_summary(records: List[Dict]) -> str:
    """把 layer_summary 的结果格式化为可打印的表格字符串."""
    header = f"{'Layer':<30}{'Type':<18}{'Input':<20}{'Output':<20}{'Params':>12}"
    lines = [header, "-" * len(header)]
    total = 0
    for r in records:
        lines.append(
            f"{r['name']:<30}{r['type']:<18}"
            f"{str(r['in_shape']):<20}{str(r['out_shape']):<20}"
            f"{r['params']:>12,}"
        )
        total += r["params"]
    lines.append("-" * len(header))
    lines.append(f"Total params: {total:,}  ({total/1e6:.2f} M)")
    return "\n".join(lines)


@torch.no_grad()
def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[float]:
    """计算批次 Top-k 准确率 (百分比)."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / batch_size * 100.0).item())
    return res
