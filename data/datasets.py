"""
数据集加载与数据增强
支持 CIFAR-100 / STL-10 / Fashion-MNIST
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 各数据集对应的均值/方差 (基于训练集统计, 近似 ImageNet 量级)
_STATS = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "stl10":    ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
}


class Cutout:
    """Cutout 数据增强: 在图像上随机挖去一个方块, 提升模型鲁棒性.

    DeVries & Taylor, Improved Regularization of CNNs with Cutout, 2017
    """

    def __init__(self, n_holes: int = 1, length: int = 8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), dtype=img.dtype)
        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0
        return img * mask.unsqueeze(0)


def _build_transforms(dataset: str, train: bool, input_size: int, aug_strategy: str):
    """构建 transform 流水线.

    aug_strategy:
        - "basic"  : 仅 RandomHorizontalFlip + 标准化
        - "strong" : RandomCrop + HorizontalFlip + ColorJitter + Cutout
    """
    mean, std = _STATS[dataset]
    ops = []

    # resize 到目标输入尺寸 (默认为数据集原始尺寸, 224 用于进阶实验)
    orig_size = {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[dataset]
    if input_size != orig_size:
        ops.append(transforms.Resize(input_size))

    if train:
        if aug_strategy == "strong":
            # 组合增强
            ops.append(transforms.RandomCrop(input_size, padding=4))
            ops.append(transforms.RandomHorizontalFlip())
            if dataset != "fashion_mnist":  # 灰度图跳过色彩抖动
                ops.append(transforms.ColorJitter(0.2, 0.2, 0.2))
        else:
            # 基础增强
            ops.append(transforms.RandomHorizontalFlip())

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean, std))

    if train and aug_strategy == "strong":
        # Cutout 作用于 tensor, 放在最后
        ops.append(Cutout(n_holes=1, length=max(8, input_size // 4)))

    return transforms.Compose(ops)


def get_dataloaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
    input_size: int = None,
    aug_strategy: str = "basic",
):
    """返回 (train_loader, test_loader).

    自动处理下载与数据增强.
    """
    dataset = dataset.lower()
    if input_size is None:
        input_size = {"cifar100": 32, "stl10": 96, "fashion_mnist": 28}[dataset]

    train_tf = _build_transforms(dataset, True, input_size, aug_strategy)
    test_tf = _build_transforms(dataset, False, input_size, aug_strategy)

    if dataset == "cifar100":
        train_set = datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
    elif dataset == "stl10":
        train_set = datasets.STL10(data_root, split="train", download=True, transform=train_tf)
        test_set = datasets.STL10(data_root, split="test", download=True, transform=test_tf)
    elif dataset == "fashion_mnist":
        train_set = datasets.FashionMNIST(data_root, train=True, download=True, transform=train_tf)
        test_set = datasets.FashionMNIST(data_root, train=False, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def get_class_names(dataset: str):
    """返回类别名称列表 (用于混淆矩阵与预测可视化)."""
    dataset = dataset.lower()
    if dataset == "fashion_mnist":
        return ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    if dataset == "stl10":
        return ["airplane", "bird", "car", "cat", "deer",
                "dog", "horse", "monkey", "ship", "truck"]
    # CIFAR-100 的 100 个 fine label
    return [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
        "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
        "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
        "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
    ]
