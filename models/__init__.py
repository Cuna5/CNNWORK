from .alexnet import AlexNet
from .vgg16 import VGG16


def build_model(name: str, num_classes: int, in_channels: int = 3, **kwargs):
    """根据名称构建模型."""
    name = name.lower()
    if name == "alexnet":
        return AlexNet(num_classes=num_classes, in_channels=in_channels, **kwargs)
    elif name == "vgg16":
        return VGG16(num_classes=num_classes, in_channels=in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
