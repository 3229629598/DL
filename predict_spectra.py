"""对一个新的气味目录进行光谱 CNN 推理。

输入目录应包含与训练数据相同命名的材料 txt；none 空白光谱从配置的
raw_dir/none 读取。程序会生成 50 个重复样本，对 50 个 softmax 概率取平均，
输出气味类别和 top-k 概率。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from cnnlab.config import load_config
from cnnlab.engine import load_checkpoint
from cnnlab.model import build_model
from cnnlab.spectra import make_odor_images
from cnnlab.utils import resolve_device, resolve_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="光谱气味 CNN 预测")
    parser.add_argument("--config", default=None, help="可选；不填时使用断点中的配置")
    parser.add_argument("--checkpoint", default=None, help="可选；也可在 predict.checkpoint 配置")
    parser.add_argument("--odor-dir", "--source", dest="odor_dir", default=None, help="待预测气味目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    args = parse_args(argv)
    preview_config = None
    checkpoint_value = args.checkpoint
    if checkpoint_value is None:
        if args.config is None:
            raise ValueError("必须提供 --checkpoint，或在配置 predict.checkpoint 中指定。")
        preview_config = load_config(args.config, overrides=list(args.set or []))
        checkpoint_value = (preview_config.get("predict") or {}).get("checkpoint")
        if not checkpoint_value:
            raise ValueError("配置中没有 predict.checkpoint。")
    checkpoint_path = resolve_path(checkpoint_value)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"断点不存在：{checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    config = (
        preview_config
        if preview_config is not None
        else (load_config(args.config, overrides=list(args.set or [])) if args.config else dict(checkpoint["config"]))
    )
    predict_cfg = config.get("predict") or {}
    odor_value = args.odor_dir or predict_cfg.get("source") or predict_cfg.get("spectra_dir")
    if not odor_value:
        raise ValueError("必须提供 --odor-dir，或在 predict.source 配置气味目录。")
    odor_dir = resolve_path(odor_value)
    if odor_dir is None:
        raise ValueError("气味目录路径无效。")

    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    input_shape = tuple(int(value) for value in checkpoint["input_shape"])
    num_classes = int(checkpoint["num_classes"])
    class_names = [str(value) for value in checkpoint.get("class_names", [])]
    if len(class_names) != num_classes:
        class_names = [f"class_{index}" for index in range(num_classes)]

    model = build_model(config, input_shape, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    images, meta = make_odor_images(config, odor_dir)
    if tuple(images.shape[1:]) != input_shape:
        raise ValueError(
            f"预测输入形状 {tuple(images.shape[1:])} 与断点 {input_shape} 不一致。"
        )

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(images))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(predict_cfg.get("batch_size", 32)),
        shuffle=False,
    )
    probability_sum = torch.zeros(num_classes, dtype=torch.float64)
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(device)).float()
            probability_sum += torch.softmax(logits, dim=1).double().cpu().sum(dim=0)
    probabilities = (probability_sum / len(dataset)).numpy()
    top_indices = np.argsort(-probabilities)[: max(1, min(args.topk, num_classes))]
    topk = [
        {"class": class_names[int(index)], "probability": float(probabilities[int(index)])}
        for index in top_indices
    ]

    output_path = resolve_path(args.output or predict_cfg.get("output_csv", "predictions_spectra.csv"))
    if output_path is None:
        output_path = Path("predictions_spectra.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "class", "probability"])
        for rank, item in enumerate(topk, start=1):
            writer.writerow([rank, item["class"], item["probability"]])
    result = {
        "odor_dir": str(odor_dir),
        "checkpoint": str(checkpoint_path),
        "predicted_class": topk[0]["class"],
        "confidence": topk[0]["probability"],
        "topk": topk,
        "materials": meta["materials"],
        "scan_count": meta["scan_count"],
    }
    (output_path.parent / "prediction_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"预测：{topk[0]['class']}  置信度：{topk[0]['probability'] * 100:.2f}%  "
        f"（融合 {len(dataset)} 条扫描）"
    )
    for rank, item in enumerate(topk, start=1):
        print(f"  {rank}. {item['class']}: {item['probability'] * 100:.2f}%")
    print(f"结果已保存：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

