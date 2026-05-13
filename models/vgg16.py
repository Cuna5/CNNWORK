"""
VGG16 从零实现
论文参考: Simonyan & Zisserman, Very Deep CNNs for Large-Scale Image Recognition, ICLR 2015
特点:
  - 全部使用 3x3 卷积 + 2x2 最大池化, 网络更深但结构规整
  - 支持两种显存优化: 将 4096 -> 1024, 或使用全局平均池化 (GAP)
"""
import torch
import torch.nn as nn


# VGG16 配置: 数字表示 Conv 输出通道, "M" 表示 MaxPool
VGG16_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


def _make_layers(cfg, in_channels: int, batch_norm: bool = True):
    """按配置构建特征提取层."""
    layers = []
    c_in = in_channels
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(c_in, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            c_in = v
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    """VGG16.

    参数:
        num_classes: 分类类别数
        in_channels: 输入通道数 (Fashion-MNIST=1, 其余=3)
        fc_dim: 全连接隐层维度 (默认 1024, 原版为 4096, 降低以节省显存)
        use_gap: 是否用全局平均池化替代传统全连接, 大幅降低参数量
        batch_norm: 是否在每个卷积后加 BN (加速收敛, 小分辨率推荐开启)
        dropout: 全连接层 Dropout 概率
    """

    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3,
        fc_dim: int = 1024,
        use_gap: bool = False,
        batch_norm: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.use_gap = use_gap
        self.features = _make_layers(VGG16_CFG, in_channels, batch_norm=batch_norm)

        # 小分辨率输入经过 5 次池化后尺寸可能变为 1x1, 用 AdaptiveAvgPool 做兜底
        # 非 GAP 模式统一拉到 2x2, 降低后续全连接的输入维度
        self.avgpool = nn.AdaptiveAvgPool2d(1 if use_gap else 2)

        if use_gap:
            # 方案 A: 全局平均池化 + 单一线性层
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )
        else:
            # 方案 B: 降维后的全连接 (默认 1024, 远小于原版 4096)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 2 * 2, fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fc_dim, num_classes),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
