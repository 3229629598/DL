"""在验证集或测试集上评估指定断点。"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import torch

from cnnlab.config import load_config, save_config_snapshot
from cnnlab.datasets import build_dataloaders, build_datasets, dataset_targets
from cnnlab.engine import evaluate_loader, load_checkpoint, save_evaluation_artifacts
from cnnlab.losses import build_loss
from cnnlab.model import build_model
from cnnlab.utils import (
    PROJECT_ROOT,
    configure_matplotlib_cache,
    create_run_dir,
    resolve_device,
    resolve_path,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="配置驱动的 CNN 评估脚本")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", required=True, help="训练保存的 .pth 断点")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="评估哪个数据划分")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="临时覆盖配置")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    args = parse_args(argv)
    config = load_config(args.config, overrides=list(args.set or []))
    runtime_cfg = config["runtime"]
    seed = int(runtime_cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(runtime_cfg.get("deterministic", False)))

    checkpoint_path = resolve_path(args.checkpoint)
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"断点不存在：{checkpoint_path}")

    output_base = resolve_path(runtime_cfg.get("output_dir", "outputs/runs"), PROJECT_ROOT)
    assert output_base is not None
    run_dir = create_run_dir(
        output_base,
        prefix=f"eval_{args.split}_{config['dataset']['name']}",
        run_name=checkpoint_path.stem,
        unique=True,
    )
    logger = setup_logger(
        run_dir,
        log_file="evaluate.log",
        console_level=str(config["logging"].get("console_level", "INFO")),
        file_level="DEBUG",
    )
    configure_matplotlib_cache(run_dir / "matplotlib_cache")
    save_config_snapshot(config, run_dir)
    device = resolve_device(str(runtime_cfg.get("device", "auto")), logger)
    checkpoint = load_checkpoint(checkpoint_path, device)

    bundle = build_datasets(config, seed=seed, logger=logger)
    for message in bundle.warnings:
        logger.warning(message)

    checkpoint_input = tuple(int(dim) for dim in checkpoint.get("input_shape", []))
    if checkpoint_input and checkpoint_input != bundle.input_shape:
        raise ValueError(
            f"断点输入形状 {checkpoint_input} 与当前数据形状 {bundle.input_shape} 不一致，"
            "请使用训练时相同的 dataset/transforms 配置。"
        )
    if int(checkpoint.get("num_classes", bundle.num_classes)) != bundle.num_classes:
        raise ValueError("断点类别数与当前数据集不一致。")

    model = build_model(config, bundle.input_shape, bundle.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    pin_memory = bool(config["dataset"].get("pin_memory", device.type == "cuda"))
    _, val_loader, test_loader = build_dataloaders(
        bundle,
        config["dataset"],
        seed=seed,
        pin_memory=pin_memory,
    )
    loader = test_loader if args.split == "test" else val_loader
    if loader is None:
        raise ValueError("当前数据集没有测试集，请改用 --split val。")

    train_targets = dataset_targets(bundle.train)
    criterion, _ = build_loss(config["loss"], bundle.num_classes, train_targets, device)
    amp_cfg = config["train"].get("amp") or {}
    amp_dtype_name = str(amp_cfg.get("dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if "bf" in amp_dtype_name else torch.float16
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"

    logger.info("评估 %s，断点：%s", args.split, checkpoint_path)
    metrics = evaluate_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        num_classes=bundle.num_classes,
        topk=tuple(int(k) for k in config["evaluation"].get("topk", [1, 5])),
        binary_threshold=float(config["evaluation"].get("binary_threshold", 0.5)),
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        logger=logger,
    )
    save_evaluation_artifacts(metrics, bundle.class_names, run_dir, args.split)
    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "metrics": metrics,
            "class_names": bundle.class_names,
        },
        run_dir / "evaluation_result.json",
    )
    logger.info(
        "评估完成 | loss %.4f | acc %.2f%% | macro_f1 %.2f%% | 指标目录：%s",
        metrics["loss"],
        metrics["accuracy"] * 100,
        metrics["f1_macro"] * 100,
        run_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

