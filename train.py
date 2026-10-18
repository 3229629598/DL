"""训练入口：只负责读取配置、组装组件并调用 Trainer。

所有训练参数都来自 YAML/命令行覆盖项，不在本文件中硬编码学习率、
网络层数或数据增强参数。
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import torch

from cnnlab.config import load_config, save_config_snapshot
from cnnlab.datasets import build_dataloaders, build_datasets
from cnnlab.engine import Trainer
from cnnlab.model import build_model
from cnnlab.utils import (
    PROJECT_ROOT,
    configure_matplotlib_cache,
    create_run_dir,
    describe_environment,
    format_seconds,
    resolve_device,
    resolve_path,
    save_json,
    set_seed,
    setup_logger,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="配置驱动的通用 CNN 训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="场景配置文件路径，例如 configs/mnist_lenet.yaml",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="临时覆盖配置，例如 --set train.epochs=1 dataset.batch_size=32",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只构建模型并执行一个 batch 的前向/反向，不写训练权重",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # 避免 Windows 控制台默认编码与中文日志冲突；不支持 reconfigure 的环境可忽略。
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    args = parse_args(argv)
    overrides = list(args.set or [])
    if args.dry_run:
        overrides.append("runtime.dry_run=true")
    config = load_config(args.config, overrides=overrides)

    runtime_cfg = config["runtime"]
    seed = int(runtime_cfg.get("seed", 42))
    set_seed(
        seed,
        deterministic=bool(runtime_cfg.get("deterministic", False)),
        benchmark=bool(runtime_cfg.get("benchmark", True)),
    )
    if runtime_cfg.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(runtime_cfg["float32_matmul_precision"]))

    output_base = resolve_path(runtime_cfg.get("output_dir", "outputs/runs"), PROJECT_ROOT)
    assert output_base is not None
    dataset_name = str(config["dataset"]["name"])
    run_dir = create_run_dir(
        output_base,
        prefix=dataset_name,
        run_name=config["logging"].get("run_name"),
        unique=bool(config["logging"].get("unique_run_dir", True)),
    )
    logger = setup_logger(
        run_dir,
        log_file=str(config["logging"].get("log_file", "train.log")),
        console_level=str(config["logging"].get("console_level", "INFO")),
        file_level=str(config["logging"].get("file_level", "DEBUG")),
    )
    configure_matplotlib_cache(run_dir / "matplotlib_cache")
    save_config_snapshot(config, run_dir)
    save_json(describe_environment(), run_dir / "environment.json")

    logger.info("=" * 80)
    logger.info("项目目录：%s", PROJECT_ROOT)
    logger.info("场景配置：%s", config["_meta"]["config_path"])
    logger.info("运行目录：%s", run_dir)
    logger.info("随机种子：%d，dry_run=%s", seed, bool(runtime_cfg.get("dry_run", False)))
    if args.set:
        logger.info("命令行覆盖项：%s", args.set)
    logger.info("=" * 80)

    device = resolve_device(str(runtime_cfg.get("device", "auto")), logger)

    logger.info("正在构建数据集……")
    bundle = build_datasets(config, seed=seed, logger=logger)
    for message in bundle.warnings:
        logger.warning(message)
    logger.info(
        "数据集：train=%d, val=%d, test=%d，输入形状=%s，类别数=%d",
        bundle.train_size,
        bundle.val_size,
        bundle.test_size,
        bundle.input_shape,
        bundle.num_classes,
    )

    # 模型输入形状以真实数据变换后的形状为准，避免配置与实际张量尺寸不一致。
    model = build_model(config, bundle.input_shape, bundle.num_classes).to(device)
    summary = model.architecture_summary()
    (run_dir / "model_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("\n%s", summary)

    pin_memory = bool(config["dataset"].get("pin_memory", device.type == "cuda"))
    train_loader, val_loader, test_loader = build_dataloaders(
        bundle,
        config["dataset"],
        seed=seed,
        pin_memory=pin_memory,
    )

    trainer = Trainer(
        config=config,
        model=model,
        bundle=bundle,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        run_dir=run_dir,
        logger=logger,
    )

    if bool(runtime_cfg.get("dry_run", False)):
        # dry-run 用来快速发现数据形状、层配置和反向传播问题，不启动完整训练。
        inputs, targets = next(iter(train_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.train()
        with torch.amp.autocast(
            device_type=device.type,
            dtype=trainer.amp_dtype,
            enabled=trainer.amp_enabled,
        ):
            logits = model(inputs)
            loss = trainer.criterion(
                logits,
                targets.float().unsqueeze(1) if bundle.num_classes == 1 else targets.long(),
            )
        loss.backward()
        logger.info(
            "dry-run 成功：输入 %s -> logits %s，loss=%.6f，反向传播已执行。",
            tuple(inputs.shape),
            tuple(logits.shape),
            float(loss.item()),
        )
        return 0

    result = trainer.fit()
    logger.info("=" * 80)
    logger.info("训练完成：最佳 %s=%.6f（epoch=%d）", result["monitor"], result["best_metric"], result["best_epoch"])
    logger.info("训练总耗时：%s", format_seconds(result["training_seconds"]))
    logger.info("最佳权重：%s", result["best_checkpoint"])
    logger.info("最后权重：%s", result["last_checkpoint"])
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

