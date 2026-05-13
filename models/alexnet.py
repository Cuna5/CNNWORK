"""
AlexNet 从零实现
针对小分辨率数据集 (CIFAR-100 32x32 / Fashion-MNIST 28x28) 做了适配:
  - 首层卷积核 11x11 -> 3x3, stride=1, padding=1
  - 池化层数量对应减少
  - 在全连接层前使用 AdaptiveAvgPool2d 兼容不同输入尺寸
参考原论文: Krizhevsky et al., ImageNet Classification with Deep CNNs, NeurIPS 2012
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """适配小分辨率的 AlexNet.

    原始 AlexNet 针对 224x224 输入. 本实现保持"5 卷积 + 3 全连接"的骨架,
    仅调整首层卷积核/步长以适配 32x32 或 28x28 的小图.
    """

    def __init__(self, num_classes: int = 100, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()

        # ---------- 特征提取 ----------
        # 输入:  (N, in_channels, H, W)   H=W=32 时的尺寸变化见 analyze.py
        self.features = nn.Sequential(
            # Conv1: kernel=3, pad=1 -> 通道 64
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            # -> H/2

            # Conv2: kernel=3, pad=1 -> 通道 192
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            # -> H/4

            # Conv3: kernel=3, pad=1 -> 通道 384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: kernel=3, pad=1 -> 通道 256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: kernel=3, pad=1 -> 通道 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            # -> H/8
        )

        # 自适应池化, 兼容 32/96/224 等不同输入
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # ---------- 分类头 ----------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化, 加快收敛并缓解梯度消失."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
