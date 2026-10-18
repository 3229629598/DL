"""数据集、数据增强与 DataLoader 构建。

支持：
- torchvision 内置数据集：MNIST/FashionMNIST/KMNIST/CIFAR10/CIFAR100。
- ImageFolder 自定义数据集：train/val/test 目录，或从 train 中按比例划分验证集。
- 合成数据集：不需要下载任何数据，适合快速检查配置和代码是否正常。

数据集层最终返回统一的 DataBundle，训练、评估、预测脚本只依赖这个结构。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .utils import PROJECT_ROOT, resolve_path, seed_worker


@dataclass
class DataBundle:
    """一次实验所需的数据集集合与元信息。"""

    train: Dataset
    val: Dataset
    test: Dataset | None
    class_names: list[str]
    input_shape: tuple[int, int, int]
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    train_size: int
    val_size: int
    test_size: int
    warnings: list[str] = field(default_factory=list)


class IndexedTransformDataset(Dataset):
    """在已有数据集的索引子集上应用指定 transform。

    需要这个包装器的原因：同一个原始训练集要划分成训练/验证两份，
    两份必须使用不同的数据增强。直接对 Subset 设置 transform 会共享同一对象，
    无法分别配置。
    """

    def __init__(self, base: Dataset, indices: Sequence[int], transform: Callable | None) -> None:
        self.base = base
        self.indices = [int(index) for index in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class SyntheticPatternDataset(Dataset):
    """可复现的彩色/灰度方块图案分类数据集。

    每个类别对应 4x4 网格中的一个位置，并带有类别相关颜色。
    数据很容易学习，适合验证网络能否正常前向、反向和收敛，
    但不应把在该数据集上的准确率当作真实视觉任务的性能。
    """

    def __init__(
        self,
        num_samples: int,
        image_size: tuple[int, int],
        channels: int,
        num_classes: int,
        noise_std: float,
        seed: int,
        split: str,
    ) -> None:
        if num_samples < 1:
            raise ValueError("synthetic.num_samples 必须大于 0。")
        if channels < 1:
            raise ValueError("synthetic.channels 必须大于 0。")
        if num_classes < 2:
            raise ValueError("synthetic.num_classes 必须至少为 2。")

        self.num_samples = int(num_samples)
        self.image_size = tuple(int(size) for size in image_size)
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.noise_std = float(noise_std)

        # 不同 split 使用不同随机种子，避免验证/测试样本与训练样本完全相同。
        split_offset = {"train": 0, "val": 1, "test": 2}.get(split, 3)
        generator = torch.Generator().manual_seed(int(seed) + split_offset * 104_729)
        height, width = self.image_size
        images = torch.randn(
            (self.num_samples, self.channels, height, width),
            generator=generator,
        ) * self.noise_std
        # torch.arange 不接受 generator 参数，但该序列本身完全确定，无需额外随机性。
        labels = torch.arange(self.num_samples, dtype=torch.long) % self.num_classes

        # 用类别决定方块位置和颜色；block 大小至少 2 像素。
        block = max(2, min(height, width) // 3)
        for index in range(self.num_samples):
            label = int(labels[index])
            row = label // 4
            column = label % 4
            center_y = int((row + 0.5) * height / 4)
            center_x = int((column + 0.5) * width / 4)
            y0 = max(0, min(height - block, center_y - block // 2))
            y1 = min(height, y0 + block)
            x0 = max(0, min(width - block, center_x - block // 2))
            x1 = min(width, x0 + block)

            # 颜色来自类别索引的循环映射，并叠加一个全局亮度背景。
            # 让 10 个类别具备可区分的通道均值；这样即使网络使用 global_avgpool，
            # 也能在少量 epoch 内学到类别差异，适合作为流程冒烟测试。
            colors = torch.tensor(
                [
                    0.25 + 0.06 * label,
                    0.20 + 0.06 * ((label * 3) % 7),
                    0.20 + 0.06 * ((label * 5 + 1) % 7),
                ],
                dtype=torch.float32,
            )
            # 整体色调也随类别变化，保证即使是 global_avgpool 网络也能快速分离类别。
            # 这使合成数据成为一个“流程正确性”测试，而不是困难的视觉 benchmark。
            tint = torch.tensor(
                [float(colors[channel % 3]) for channel in range(self.channels)],
                dtype=torch.float32,
            ).view(self.channels, 1, 1)
            images[index] += 0.35 * tint
            for channel in range(self.channels):
                images[index, channel, y0:y1, x0:x1] += colors[channel % 3] * 1.5

        self.images = images.clamp_(0.0, 1.0)
        self.labels = labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        return self.images[index], int(self.labels[index])


# 内置数据集的默认统计量。用户在配置中设置 normalize.mode=custom 可覆盖。
BUILTIN_DATASETS: dict[str, dict[str, Any]] = {
    "mnist": {
        "class_count": 10,
        "channels": 1,
        "image_size": 28,
        "mean": (0.1307,),
        "std": (0.3081,),
        "classes": [str(index) for index in range(10)],
    },
    "fashion_mnist": {
        "class_count": 10,
        "channels": 1,
        "image_size": 28,
        "mean": (0.2860,),
        "std": (0.3530,),
        "classes": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    },
    "kmnist": {
        "class_count": 10,
        "channels": 1,
        "image_size": 28,
        "mean": (0.1918,),
        "std": (0.3483,),
        "classes": [f"class_{index}" for index in range(10)],
    },
    "cifar10": {
        "class_count": 10,
        "channels": 3,
        "image_size": 32,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "classes": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    },
    "cifar100": {
        "class_count": 100,
        "channels": 3,
        "image_size": 32,
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "classes": [f"class_{index:03d}" for index in range(100)],
    },
}


def _pair_size(value: Any, name: str) -> tuple[int, int]:
    """把 image_size 之类的配置解析成 (高, 宽)。"""

    if isinstance(value, int) and not isinstance(value, bool):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) in {1, 2}:
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    raise ValueError(f"{name} 必须是整数或长度 1/2 的列表，当前为 {value!r}。")


def _enabled(spec: Any) -> bool:
    """统一判断某个增强选项是否启用。

    支持：
    - true/false
    - 数字（大于 0 即启用）
    - 字典中的 enabled 字段
    """

    if isinstance(spec, Mapping):
        return bool(spec.get("enabled", True))
    if isinstance(spec, bool):
        return spec
    if isinstance(spec, (int, float)):
        return spec > 0
    return spec is not None


def build_transforms(
    spec: Mapping[str, Any] | None,
    train: bool,
    mean: Sequence[float],
    std: Sequence[float],
    image_size: tuple[int, int],
) -> transforms.Compose:
    """按配置组装 torchvision 变换流水线。

    执行顺序通常是：
    AutoAugment -> 裁剪/缩放 -> 翻转/旋转/仿射 -> 颜色变换 -> 灰度/模糊
    -> ToTensor -> Normalize -> RandomErasing。

    每个选项都先从配置读取，未出现时使用安全默认值（不启用）。
    """

    spec = dict(spec or {})
    operations: list[Callable] = []

    auto_augment = spec.get("auto_augment")
    if auto_augment:
        try:
            policy = transforms.AutoAugmentPolicy(str(auto_augment).upper())
        except ValueError as exc:
            raise ValueError(
                f"auto_augment='{auto_augment}' 不支持，可选 imagenet/cifar10/svhn。"
            ) from exc
        operations.append(transforms.AutoAugment(policy=policy))

    # Resize 放在裁剪之前，确保后续随机裁剪有稳定的输入尺寸。
    resize = spec.get("resize")
    if resize:
        size = _pair_size(resize, "transforms.resize")
        operations.append(transforms.Resize(size))

    random_resized_crop = spec.get("random_resized_crop") or {}
    if train and _enabled(random_resized_crop):
        crop_size = _pair_size(random_resized_crop.get("size", image_size), "random_resized_crop.size")
        operations.append(
            transforms.RandomResizedCrop(
                size=crop_size,
                scale=tuple(random_resized_crop.get("scale", (0.8, 1.0))),
                ratio=tuple(random_resized_crop.get("ratio", (0.75, 1.333333))),
            )
        )

    random_crop = spec.get("random_crop") or {}
    if train and _enabled(random_crop):
        crop_size = _pair_size(random_crop.get("size", image_size), "random_crop.size")
        padding = int(random_crop.get("padding", 0))
        padding_mode = str(random_crop.get("padding_mode", "constant"))
        fill = random_crop.get("fill", 0)
        operations.append(
            transforms.RandomCrop(
                size=crop_size,
                padding=padding,
                padding_mode=padding_mode,
                fill=fill,
            )
        )

    center_crop = spec.get("center_crop")
    if center_crop:
        operations.append(transforms.CenterCrop(_pair_size(center_crop, "center_crop")))

    horizontal_flip = float(spec.get("random_horizontal_flip", 0.0))
    if train and horizontal_flip > 0:
        operations.append(transforms.RandomHorizontalFlip(p=horizontal_flip))

    vertical_flip = float(spec.get("random_vertical_flip", 0.0))
    if train and vertical_flip > 0:
        operations.append(transforms.RandomVerticalFlip(p=vertical_flip))

    rotation = spec.get("random_rotation")
    if train and rotation:
        if isinstance(rotation, Mapping):
            degrees = float(rotation.get("degrees", 0))
            interpolation = getattr(
                transforms.InterpolationMode,
                str(rotation.get("interpolation", "NEAREST")).upper(),
            )
            operations.append(transforms.RandomRotation(degrees=degrees, interpolation=interpolation))
        else:
            operations.append(transforms.RandomRotation(degrees=float(rotation)))

    affine = spec.get("random_affine") or {}
    if train and _enabled(affine):
        operations.append(
            transforms.RandomAffine(
                degrees=float(affine.get("degrees", 0)),
                translate=tuple(affine.get("translate", (0.0, 0.0))),
                scale=tuple(affine.get("scale", (1.0, 1.0))),
                shear=float(affine.get("shear", 0.0)),
                interpolation=getattr(
                    transforms.InterpolationMode,
                    str(affine.get("interpolation", "NEAREST")).upper(),
                ),
                fill=affine.get("fill", 0),
            )
        )

    color_jitter = spec.get("color_jitter") or {}
    if train and _enabled(color_jitter):
        jitter = transforms.ColorJitter(
            brightness=float(color_jitter.get("brightness", 0)),
            contrast=float(color_jitter.get("contrast", 0)),
            saturation=float(color_jitter.get("saturation", 0)),
            hue=float(color_jitter.get("hue", 0)),
        )
        probability = float(color_jitter.get("p", 1.0))
        operations.append(transforms.RandomApply([jitter], p=probability))

    grayscale = spec.get("grayscale") or {}
    if _enabled(grayscale):
        output_channels = int(grayscale.get("num_output_channels", 3))
        if train:
            operations.append(
                transforms.RandomGrayscale(
                    p=float(grayscale.get("p", 0.1)),
                    num_output_channels=output_channels,
                )
            )
        else:
            operations.append(transforms.Grayscale(num_output_channels=output_channels))

    blur = spec.get("gaussian_blur") or {}
    if train and _enabled(blur):
        operations.append(
            transforms.GaussianBlur(
                kernel_size=int(blur.get("kernel_size", 3)),
                sigma=tuple(blur.get("sigma", (0.1, 2.0))),
            )
        )

    if bool(spec.get("to_tensor", True)):
        operations.append(transforms.ToTensor())
    else:
        raise ValueError(
            "transforms.to_tensor=false 会让 DataLoader 返回 PIL 图像而无法训练；"
            "请保持为 true，若需自定义张量转换请扩展 build_transforms。"
        )

    normalize = spec.get("normalize") or {}
    normalize_mode = str(normalize.get("mode", "auto")).lower()
    if normalize_mode == "custom":
        custom_mean = tuple(float(value) for value in (normalize.get("mean") or mean))
        custom_std = tuple(float(value) for value in (normalize.get("std") or std))
        operations.append(transforms.Normalize(custom_mean, custom_std))
        used_mean, used_std = custom_mean, custom_std
    elif normalize_mode == "auto":
        used_mean, used_std = tuple(float(value) for value in mean), tuple(float(value) for value in std)
        operations.append(transforms.Normalize(used_mean, used_std))
    elif normalize_mode == "none":
        used_mean, used_std = (0.0,) * len(mean), (1.0,) * len(mean)
    else:
        raise ValueError("normalize.mode 可选 auto/custom/none。")

    erasing = spec.get("random_erasing") or {}
    if train and _enabled(erasing):
        operations.append(
            transforms.RandomErasing(
                p=float(erasing.get("p", 0.25)),
                scale=tuple(erasing.get("scale", (0.02, 0.2))),
                ratio=tuple(erasing.get("ratio", (0.3, 3.3))),
                value=float(erasing.get("value", 0.0)),
                inplace=bool(erasing.get("inplace", False)),
            )
        )

    # 记录实际使用的均值方差，便于预测脚本还原同一套预处理。
    transform = transforms.Compose(operations)
    transform._cnnlab_mean = used_mean  # type: ignore[attr-defined]
    transform._cnnlab_std = used_std  # type: ignore[attr-defined]
    return transform


def _resolve_dataset_mean_std(
    dataset_cfg: Mapping[str, Any],
    default_mean: Sequence[float],
    default_std: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """解析数据集级别的均值方差配置。

    normalize.mode=custom 时，优先使用数据集级 mean/std；否则使用内置统计量。
    这样验证集和训练集不会因为分别读取 transform 配置而出现不一致。
    """

    transform_cfg = dataset_cfg.get("transforms") or {}
    train_normalize = (transform_cfg.get("train") or {}).get("normalize") or {}
    val_normalize = (transform_cfg.get("val") or {}).get("normalize") or {}
    train_mode = str(train_normalize.get("mode", "auto")).lower()
    val_mode = str(val_normalize.get("mode", "auto")).lower()
    if "custom" in {train_mode, val_mode}:
        mean = tuple(float(value) for value in (dataset_cfg.get("mean") or default_mean))
        std = tuple(float(value) for value in (dataset_cfg.get("std") or default_std))
        return mean, std
    return tuple(float(value) for value in default_mean), tuple(float(value) for value in default_std)


def _build_builtin(
    name: str,
    root: Path,
    train: bool,
    transform: Callable | None,
    download: bool,
) -> Dataset:
    """延迟导入 torchvision.datasets，减少无关脚本的启动开销。"""

    from torchvision import datasets

    mapping = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "kmnist": datasets.KMNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    cls = mapping[name]
    return cls(root=str(root), train=bool(train), transform=transform, download=bool(download))


def _check_image_folder_classes(
    train_dataset: Dataset,
    other_dataset: Dataset,
    split_name: str,
) -> None:
    """确保验证/测试目录的类别映射与训练目录完全一致。"""

    train_classes = getattr(train_dataset, "classes", None)
    other_classes = getattr(other_dataset, "classes", None)
    if train_classes != other_classes:
        raise ValueError(
            f"{split_name} 目录的类别与训练目录不一致。\n"
            f"训练类别：{train_classes}\n{split_name} 类别：{other_classes}\n"
            "请保证每个子目录名称及其排序完全一致。"
        )


def _validate_dummy_transform(
    transform: Callable,
    image_size: tuple[int, int],
    channels: int,
    name: str,
) -> tuple[int, int, int]:
    """用空白图片跑一遍变换，提前发现最终尺寸与配置不符的问题。"""

    mode = "L" if channels == 1 else "RGB"
    dummy = Image.new(mode, (image_size[1], image_size[0]))
    output = transform(dummy)
    if not isinstance(output, torch.Tensor) or output.dim() != 3:
        raise ValueError(f"{name} 的输出应为 [C,H,W] 张量，实际为 {type(output)}。")
    return tuple(int(size) for size in output.shape)  # type: ignore[return-value]


def build_datasets(
    config: Mapping[str, Any],
    seed: int,
) -> DataBundle:
    """根据 dataset 配置构建训练、验证、测试数据集。"""

    dataset_cfg = config["dataset"]
    name = str(dataset_cfg["name"]).lower()
    root = resolve_path(dataset_cfg.get("root", "data"), PROJECT_ROOT)
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)

    image_size = _pair_size(dataset_cfg.get("image_size", 32), "dataset.image_size")
    transforms_cfg = dataset_cfg.get("transforms") or {}
    warnings: list[str] = []

    if name == "synthetic":
        synthetic_cfg = dataset_cfg.get("synthetic") or {}
        # 合成数据优先使用 synthetic.channels，便于只改一个字段切换灰度/彩色任务。
        channels = int(synthetic_cfg.get("channels", dataset_cfg.get("in_channels", 3)))
        num_classes = int(dataset_cfg.get("num_classes") or synthetic_cfg.get("num_classes", 10))
        image_size = _pair_size(synthetic_cfg.get("image_size", image_size), "synthetic.image_size")
        train_samples = int(synthetic_cfg.get("train_samples", 512))
        val_samples = int(synthetic_cfg.get("val_samples", 128))
        test_samples = int(synthetic_cfg.get("test_samples", 128))
        noise_std = float(synthetic_cfg.get("noise_std", 0.15))

        train = SyntheticPatternDataset(
            train_samples, image_size, channels, num_classes, noise_std, seed, "train"
        )
        val = SyntheticPatternDataset(
            val_samples, image_size, channels, num_classes, noise_std, seed, "val"
        )
        test = SyntheticPatternDataset(
            test_samples, image_size, channels, num_classes, noise_std, seed, "test"
        )
        class_names = [f"class_{index}" for index in range(num_classes)]
        mean = tuple([0.5] * channels)
        std = tuple([0.5] * channels)
        return DataBundle(
            train=train,
            val=val,
            test=test,
            class_names=class_names,
            input_shape=(channels, *image_size),
            num_classes=num_classes,
            mean=mean,
            std=std,
            train_size=len(train),
            val_size=len(val),
            test_size=len(test),
            warnings=warnings,
        )

    if name in BUILTIN_DATASETS:
        meta = BUILTIN_DATASETS[name]
        channels = int(dataset_cfg.get("in_channels", meta["channels"]))
        if channels != int(meta["channels"]):
            raise ValueError(
                f"{name} 原始通道数为 {meta['channels']}，dataset.in_channels={channels} 不一致。"
            )
        num_classes = int(meta["class_count"])
        configured_classes = dataset_cfg.get("num_classes")
        if configured_classes not in {None, "auto", num_classes}:
            raise ValueError(
                f"{name} 的类别数固定为 {num_classes}，但 dataset.num_classes={configured_classes}。"
            )

        mean, std = _resolve_dataset_mean_std(dataset_cfg, meta["mean"], meta["std"])
        train_tf = build_transforms(
            transforms_cfg.get("train", {}), True, mean, std, image_size
        )
        val_tf = build_transforms(
            transforms_cfg.get("val", {}), False, mean, std, image_size
        )
        input_shape = _validate_dummy_transform(train_tf, image_size, channels, "训练变换")
        val_shape = _validate_dummy_transform(val_tf, image_size, channels, "验证变换")
        if input_shape != val_shape:
            raise ValueError(f"训练变换输出 {input_shape} 与验证变换输出 {val_shape} 不一致。")

        download = bool(dataset_cfg.get("download", True))
        # 原始数据集不附带 transform，划分索引后再分别套用训练/验证变换。
        train_base = _build_builtin(name, root, True, None, download)
        test_base = _build_builtin(name, root, False, None, download)
        class_names = list(getattr(train_base, "classes", meta["classes"]))

        val_split = float(dataset_cfg.get("val_split", 0.1))
        if not 0.0 <= val_split < 1.0:
            raise ValueError("dataset.val_split 必须在 [0,1) 范围内。")

        if val_split > 0:
            total = len(train_base)
            val_size = max(1, int(round(total * val_split)))
            train_size = total - val_size
            if train_size < 1:
                raise ValueError("验证集比例过大，训练集至少需要保留 1 个样本。")
            generator = torch.Generator().manual_seed(int(seed))
            indices = torch.randperm(total, generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train = IndexedTransformDataset(train_base, train_indices, train_tf)
            val = IndexedTransformDataset(train_base, val_indices, val_tf)
            test = IndexedTransformDataset(test_base, list(range(len(test_base))), val_tf)
        elif bool(dataset_cfg.get("use_test_as_val", False)):
            warnings.append(
                "dataset.use_test_as_val=true：验证集就是测试集，最终测试指标会偏乐观。"
            )
            train = IndexedTransformDataset(train_base, list(range(len(train_base))), train_tf)
            val = IndexedTransformDataset(test_base, list(range(len(test_base))), val_tf)
            test = IndexedTransformDataset(test_base, list(range(len(test_base))), val_tf)
        else:
            raise ValueError(
                "dataset.val_split=0 且未设置 use_test_as_val=true，无法获得验证集。"
                "请设置一个 0~1 的比例。"
            )

        return DataBundle(
            train=train,
            val=val,
            test=test,
            class_names=class_names,
            input_shape=input_shape,
            num_classes=num_classes,
            mean=mean,
            std=std,
            train_size=len(train),
            val_size=len(val),
            test_size=len(test),
            warnings=warnings,
        )

    if name == "image_folder":
        folder_cfg = dataset_cfg.get("image_folder") or {}
        train_dir = resolve_path(folder_cfg.get("train_dir"), PROJECT_ROOT)
        val_dir = resolve_path(folder_cfg.get("val_dir"), PROJECT_ROOT)
        test_dir = resolve_path(folder_cfg.get("test_dir"), PROJECT_ROOT)
        if train_dir is None or not train_dir.exists():
            raise FileNotFoundError(
                f"image_folder.train_dir 不存在：{train_dir}。目录下应按类别建立子文件夹。"
            )

        # ImageFolder 的类别由子目录名排序得到；训练创建后，以此为准检查其他划分。
        from torchvision.datasets import ImageFolder

        inferred_channels = int(dataset_cfg.get("in_channels", 3))
        mean, std = _resolve_dataset_mean_std(
            dataset_cfg,
            [0.485, 0.456, 0.406] if inferred_channels == 3 else [0.5],
            [0.229, 0.224, 0.225] if inferred_channels == 3 else [0.5],
        )
        train_tf = build_transforms(
            transforms_cfg.get("train", {}), True, mean, std, image_size
        )
        val_tf = build_transforms(
            transforms_cfg.get("val", {}), False, mean, std, image_size
        )
        input_shape = _validate_dummy_transform(train_tf, image_size, inferred_channels, "训练变换")
        val_shape = _validate_dummy_transform(val_tf, image_size, inferred_channels, "验证变换")
        if input_shape != val_shape:
            raise ValueError(f"训练变换输出 {input_shape} 与验证变换输出 {val_shape} 不一致。")

        train_base = ImageFolder(root=str(train_dir), transform=None)
        class_names = list(train_base.classes)
        num_classes = len(class_names)
        configured_classes = dataset_cfg.get("num_classes")
        if configured_classes not in {None, "auto", num_classes}:
            raise ValueError(
                f"训练目录检测到 {num_classes} 个类别，但 dataset.num_classes={configured_classes}。"
            )

        if val_dir is not None:
            val_base = ImageFolder(root=str(val_dir), transform=None)
            _check_image_folder_classes(train_base, val_base, "验证集")
            train = IndexedTransformDataset(train_base, range(len(train_base)), train_tf)
            val = IndexedTransformDataset(val_base, range(len(val_base)), val_tf)
        else:
            val_split = float(dataset_cfg.get("val_split", 0.1))
            if not 0.0 < val_split < 1.0:
                raise ValueError(
                    "未配置 image_folder.val_dir 时，dataset.val_split 必须在 (0,1) 范围内。"
                )
            total = len(train_base)
            val_size = max(1, int(round(total * val_split)))
            train_size = total - val_size
            generator = torch.Generator().manual_seed(int(seed))
            indices = torch.randperm(total, generator=generator).tolist()
            train = IndexedTransformDataset(train_base, indices[:train_size], train_tf)
            val = IndexedTransformDataset(train_base, indices[train_size:], val_tf)
            warnings.append("未配置 val_dir，已从训练目录随机划分验证集。")

        if test_dir is not None:
            if not test_dir.exists():
                raise FileNotFoundError(f"image_folder.test_dir 不存在：{test_dir}")
            test_base = ImageFolder(root=str(test_dir), transform=None)
            _check_image_folder_classes(train_base, test_base, "测试集")
            test = IndexedTransformDataset(test_base, range(len(test_base)), val_tf)
        else:
            test = None
            warnings.append("未配置 test_dir，训练结束后不会自动进行测试集评估。")

        return DataBundle(
            train=train,
            val=val,
            test=test,
            class_names=class_names,
            input_shape=input_shape,
            num_classes=num_classes,
            mean=mean,
            std=std,
            train_size=len(train),
            val_size=len(val),
            test_size=len(test) if test is not None else 0,
            warnings=warnings,
        )

    raise ValueError(f"未知数据集：{name}")


def build_dataloaders(
    bundle: DataBundle,
    dataset_cfg: Mapping[str, Any],
    seed: int,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """构建训练/验证/测试 DataLoader。

    Windows 下多进程 DataLoader 必须由脚本入口的 if __name__ == "__main__":
    保护；本项目已经遵守。若仍遇到 worker 启动问题，可把 num_workers 设为 0。
    """

    workers = int(dataset_cfg.get("num_workers", 0))
    if workers < 0:
        raise ValueError("dataset.num_workers 不能为负数。")
    common: dict[str, Any] = {
        "batch_size": int(dataset_cfg.get("batch_size", 32)),
        "num_workers": workers,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(dataset_cfg.get("persistent_workers", workers > 0)) and workers > 0,
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        common["prefetch_factor"] = int(dataset_cfg.get("prefetch_factor", 2))

    train_generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        bundle.train,
        shuffle=True,
        drop_last=bool(dataset_cfg.get("drop_last", False)),
        generator=train_generator,
        **common,
    )
    val_loader = DataLoader(
        bundle.val,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = (
        DataLoader(
            bundle.test,
            shuffle=False,
            drop_last=False,
            **common,
        )
        if bundle.test is not None
        else None
    )
    return train_loader, val_loader, test_loader


def dataset_targets(dataset: Dataset) -> list[int]:
    """尽量从各种 Dataset 包装器中提取训练标签，用于计算类别权重。"""

    if isinstance(dataset, IndexedTransformDataset):
        base_targets = dataset_targets(dataset.base)
        return [base_targets[int(index)] for index in dataset.indices]
    if isinstance(dataset, Subset):
        base_targets = dataset_targets(dataset.dataset)
        return [base_targets[int(index)] for index in dataset.indices]
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(target) for target in targets]
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return [int(label) for label in labels]
    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]
    raise TypeError(f"无法从数据集 {type(dataset).__name__} 提取标签。")

