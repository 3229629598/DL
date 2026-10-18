"""使用训练好的断点对单张图片或图片目录进行预测。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cnnlab.config import load_config
from cnnlab.datasets import build_transforms
from cnnlab.engine import load_checkpoint
from cnnlab.model import build_model
from cnnlab.utils import resolve_device, resolve_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImagePathDataset(Dataset):
    """图片路径数据集；只用于推理，不返回标签。"""

    def __init__(self, paths: Sequence[Path], transform, channels: int) -> None:
        self.paths = list(paths)
        self.transform = transform
        self.channels = channels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("L" if self.channels == 1 else "RGB")
            tensor = self.transform(image)
        return tensor, str(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="配置驱动的 CNN 图片预测脚本")
    parser.add_argument("--config", default=None, help="可选；不填时直接使用断点中保存的配置")
    parser.add_argument("--checkpoint", default=None, help="断点路径；也可在 predict.checkpoint 配置")
    parser.add_argument("--source", default=None, help="单张图片或图片目录；也可在 predict.source 配置")
    parser.add_argument("--output", default=None, help="预测 CSV 路径")
    parser.add_argument("--topk", type=int, default=None, help="返回前 k 个类别")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="临时覆盖配置")
    return parser.parse_args(argv)


def collect_images(source: Path) -> list[Path]:
    """收集单文件或目录下所有受支持的图片。"""

    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的图片扩展名：{source.suffix}")
        return [source]
    if source.is_dir():
        paths = sorted(
            path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise FileNotFoundError(f"目录中没有找到图片：{source}")
        return paths
    raise FileNotFoundError(f"source 不存在：{source}")


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    args = parse_args(argv)

    # 先确定断点路径；如果命令行没给断点，则从 --config 的 predict.checkpoint 读取。
    checkpoint_arg = args.checkpoint
    preview_config = None
    if checkpoint_arg is None:
        if args.config is None:
            raise ValueError("必须通过 --checkpoint 或 predict.checkpoint 指定断点。")
        preview_config = load_config(args.config, overrides=list(args.set or []))
        checkpoint_arg = (preview_config.get("predict") or {}).get("checkpoint")
        if not checkpoint_arg:
            raise ValueError("命令行和 predict.checkpoint 都没有指定断点。")

    # 读取断点本身时先放到 CPU，兼容只有 CPU 的机器；随后再根据配置搬到目标设备。
    initial_device = torch.device("cpu")
    checkpoint_path = resolve_path(checkpoint_arg)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"断点不存在：{checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, initial_device)
    config = (
        preview_config
        if preview_config is not None
        else (load_config(args.config, overrides=list(args.set or [])) if args.config else dict(checkpoint["config"]))
    )
    predict_cfg = config.get("predict") or {}

    source_value = args.source or predict_cfg.get("source")
    if not source_value:
        raise ValueError("必须通过 --source 或 predict.source 指定图片/目录。")
    source = resolve_path(str(source_value))
    if source is None:
        raise ValueError("source 路径无效。")

    output_value = args.output or predict_cfg.get("output_csv", "predictions.csv")
    output_path = resolve_path(str(output_value)) or Path("predictions.csv")
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    input_shape = tuple(int(dim) for dim in checkpoint["input_shape"])
    if len(input_shape) != 3:
        raise ValueError(f"断点中的 input_shape 非法：{input_shape}")
    channels, height, width = input_shape
    num_classes = int(checkpoint["num_classes"])
    class_names = list(checkpoint.get("class_names") or [f"class_{i}" for i in range(num_classes)])
    if num_classes != 1 and len(class_names) != num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]

    model = build_model(config, input_shape, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    mean = tuple(float(value) for value in checkpoint.get("mean", [0.5] * channels))
    std = tuple(float(value) for value in checkpoint.get("std", [0.5] * channels))
    val_spec = ((config.get("dataset") or {}).get("transforms") or {}).get("val", {})
    transform = build_transforms(val_spec, False, mean, std, (height, width))
    # 用空白图检查预测预处理输出是否与训练时输入形状一致。
    dummy = Image.new("L" if channels == 1 else "RGB", (width, height))
    dummy_tensor = transform(dummy)
    if tuple(dummy_tensor.shape) != input_shape:
        raise ValueError(
            f"预测预处理输出 {tuple(dummy_tensor.shape)} 与断点输入 {input_shape} 不一致；"
            "请让 predict 使用训练时的 dataset.transforms.val 配置。"
        )

    paths = collect_images(source)
    batch_size = max(1, int(predict_cfg.get("batch_size", 32)))
    loader = DataLoader(
        ImagePathDataset(paths, transform, channels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    topk = max(1, int(args.topk or predict_cfg.get("topk", 5)))
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for images, batch_paths in loader:
            logits = model(images.to(device)).float()
            if num_classes == 1:
                # 单 logit 二分类：正类概率是 sigmoid(logit)。
                positive = torch.sigmoid(logits[:, 0])
                probabilities = torch.stack([1.0 - positive, positive], dim=1)
                names = ["negative", "positive"]
                predicted = (positive >= 0.5).long()
                confidence = torch.maximum(positive, 1.0 - positive)
            else:
                probabilities = torch.softmax(logits, dim=1)
                names = class_names
                confidence, predicted = probabilities.max(dim=1)

            for index, path in enumerate(batch_paths):
                count = min(topk, probabilities.shape[1])
                values, indices = probabilities[index].topk(count)
                top_list = [
                    {"class": names[int(class_index)], "probability": float(probability)}
                    for probability, class_index in zip(values.cpu(), indices.cpu())
                ]
                predicted_index = int(predicted[index].item())
                rows.append(
                    {
                        "path": str(path),
                        "predicted_index": predicted_index,
                        "predicted_class": names[predicted_index],
                        "confidence": float(confidence[index].item()),
                        "topk": json.dumps(top_list, ensure_ascii=False),
                    }
                )
                print(
                    f"{path} -> {names[predicted_index]} "
                    f"({float(confidence[index].item()) * 100:.2f}%)"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "predicted_index", "predicted_class", "confidence", "topk"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"预测结果已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

