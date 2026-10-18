"""训练、验证、评估、断点保存与曲线绘制。

训练循环遵循标准 PyTorch 流程：
前向传播 -> 计算损失 -> 反向传播 -> 梯度裁剪 -> 优化器更新。
配置只改变各步骤的参数和启用状态，不改变自动求导本身。
"""

from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch import optim

from .datasets import DataBundle, dataset_targets
from .losses import build_loss
from .metrics import (
    MetricAccumulator,
    plot_confusion_matrix,
    save_confusion_csv,
    save_per_class_csv,
)
from .utils import (
    format_seconds,
    load_json,
    resolve_path,
    save_json,
)


HISTORY_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_accuracy",
    "train_f1_macro",
    "val_loss",
    "val_accuracy",
    "val_f1_macro",
    "best_metric",
    "grad_norm",
    "optimizer_steps",
    "skipped_batches",
    "epoch_seconds",
]


def _amp_dtype(name: str) -> torch.dtype:
    """把配置中的 AMP 数据类型转换为 torch.dtype。"""

    value = str(name).lower().replace("_", "").replace("-", "")
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError("train.amp.dtype 可选 float16/fp16/bfloat16/bf16。")
    return mapping[value]


def _build_grad_scaler(
    enabled: bool,
    init_scale: float = 65536.0,
    growth_factor: float = 2.0,
    backoff_factor: float = 0.5,
    growth_interval: int = 2000,
) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler | None:
    """创建 GradScaler，并兼容不同 PyTorch 版本的构造签名。"""

    if not enabled:
        return None
    kwargs = {
        "enabled": True,
        "init_scale": float(init_scale),
        "growth_factor": float(growth_factor),
        "backoff_factor": float(backoff_factor),
        "growth_interval": int(growth_interval),
    }
    try:
        return torch.amp.GradScaler("cuda", **kwargs)
    except TypeError:
        pass
    try:
        return torch.amp.GradScaler(device="cuda", **kwargs)
    except TypeError:
        return torch.cuda.amp.GradScaler(**kwargs)


def build_optimizer(model: nn.Module, optimizer_cfg: Mapping[str, Any]) -> optim.Optimizer:
    """构建优化器，并支持按参数名划分不同的学习率/权重衰减组。"""

    name = str(optimizer_cfg.get("name", "adamw")).lower()
    base_lr = float(optimizer_cfg.get("lr", 1e-3))
    base_weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
    named_parameters = [
        (parameter_name, parameter)
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("模型没有可训练参数，请检查 trainable/freeze 配置。")

    group_specs = list(optimizer_cfg.get("param_groups") or [])
    parameter_groups: list[dict[str, Any]] = []
    assigned: set[int] = set()

    # 自定义参数组：按正则表达式匹配参数名。第一个匹配的组生效。
    for spec in group_specs:
        patterns = [str(pattern) for pattern in (spec.get("patterns") or [])]
        parameters = []
        for parameter_name, parameter in named_parameters:
            if id(parameter) in assigned:
                continue
            if any(re.search(pattern, parameter_name) for pattern in patterns):
                parameters.append(parameter)
                assigned.add(id(parameter))
        if not parameters:
            continue
        parameter_groups.append(
            {
                "name": str(spec.get("name", "custom")),
                "params": parameters,
                "lr": base_lr * float(spec.get("lr_multiplier", 1.0)),
                "weight_decay": float(spec.get("weight_decay", base_weight_decay)),
            }
        )

    remaining = [
        (parameter_name, parameter)
        for parameter_name, parameter in named_parameters
        if id(parameter) not in assigned
    ]

    if not group_specs and bool(optimizer_cfg.get("no_weight_decay_for_bias_and_norm", True)):
        decay_parameters = []
        no_decay_parameters = []
        for parameter_name, parameter in remaining:
            lower_name = parameter_name.lower()
            if lower_name.endswith(".bias") or "norm" in lower_name or "bn" in lower_name.split(".")[-2:]:
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        if decay_parameters:
            parameter_groups.append(
                {"name": "decay", "params": decay_parameters, "lr": base_lr, "weight_decay": base_weight_decay}
            )
        if no_decay_parameters:
            parameter_groups.append(
                {"name": "no_decay", "params": no_decay_parameters, "lr": base_lr, "weight_decay": 0.0}
            )
    else:
        parameters = [parameter for _, parameter in remaining]
        if parameters:
            parameter_groups.append(
                {"name": "default", "params": parameters, "lr": base_lr, "weight_decay": base_weight_decay}
            )

    if not parameter_groups:
        raise ValueError("参数分组后没有任何参数，请检查 optimizer.param_groups 的 patterns。")

    # 这些键由项目自己处理，不能原样传给 torch.optim 构造函数。
    raw_params = {
        key: value
        for key, value in (optimizer_cfg.get("params") or {}).items()
        if key
        not in {
            "lr",
            "weight_decay",
            "param_groups",
            "no_weight_decay_for_bias_and_norm",
        }
    }
    registry = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "rmsprop": optim.RMSprop,
        "adagrad": optim.Adagrad,
        "adadelta": optim.Adadelta,
    }
    if name not in registry:
        raise ValueError(
            f"未知优化器 '{name}'，可选：{', '.join(sorted(registry))}。"
        )
    try:
        return registry[name](parameter_groups, **raw_params)
    except TypeError as exc:
        raise ValueError(f"优化器 {name} 的参数不正确：{exc}") from exc


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_cfg: Mapping[str, Any],
    epochs: int,
    steps_per_epoch: int,
) -> optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None:
    """构建学习率调度器。"""

    name = str(scheduler_cfg.get("name", "none")).lower()
    params = dict(scheduler_cfg.get("params") or {})
    if name == "none":
        return None
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(params.get("step_size", max(1, epochs // 3))),
            gamma=float(params.get("gamma", 0.1)),
        )
    if name == "multi_step":
        milestones = params.get("milestones", [max(1, epochs // 3), max(2, 2 * epochs // 3)])
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(value) for value in milestones],
            gamma=float(params.get("gamma", 0.1)),
        )
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(params.get("t_max") or max(1, epochs)),
            eta_min=float(params.get("eta_min", 0.0)),
        )
    if name == "cosine_restarts":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(params.get("t_0") or max(1, epochs // 2)),
            T_mult=int(params.get("t_mult") or 1),
            eta_min=float(params.get("eta_min", 0.0)),
        )
    if name == "plateau":
        mode = str(params.get("mode", "min"))
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(params.get("factor", 0.5)),
            patience=int(params.get("patience", 3)),
            threshold=float(params.get("threshold", 1e-4)),
            threshold_mode=str(params.get("threshold_mode", "rel")),
            cooldown=int(params.get("cooldown", 0)),
            min_lr=float(params.get("min_lr", 0.0)),
            eps=float(params.get("eps", 1e-8)),
        )
    if name == "linear":
        return optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(params.get("start_factor", 1.0)),
            end_factor=float(params.get("end_factor", 0.01)),
            total_iters=int(params.get("total_iters") or max(1, epochs)),
        )
    if name == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(params.get("gamma", 0.99)),
        )
    if name == "one_cycle":
        total_steps = max(1, int(params.get("total_steps") or epochs * steps_per_epoch))
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(params.get("max_lr", optimizer.param_groups[0]["lr"])),
            total_steps=total_steps,
            pct_start=float(params.get("pct_start", 0.3)),
            anneal_strategy=str(params.get("anneal_strategy", "cos")),
            div_factor=float(params.get("div_factor", 25.0)),
            final_div_factor=float(params.get("final_div_factor", 1e4)),
            three_phase=bool(params.get("three_phase", False)),
        )
    raise ValueError(f"未知学习率调度器 '{name}'。")


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """读取断点文件；显式关闭 weights_only 以兼容包含配置字典的检查点。"""

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"断点文件格式不正确：{path}")
    return checkpoint


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """公开的断点读取接口，评估和预测脚本复用同一套兼容逻辑。"""

    return _load_checkpoint(Path(path), device)


def _unwrap_model(model: nn.Module) -> nn.Module:
    """torch.compile 会给模型加一层 _orig_mod，保存权重时使用原始模块。"""

    return getattr(model, "_orig_mod", model)


def _prepare_targets(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """根据任务类型整理损失函数需要的 target 形状。"""

    if num_classes == 1:
        return targets.float().unsqueeze(1)
    return targets.long()


def _monitor_mode(monitor: str, configured: str) -> str:
    """确定监控指标的优化方向。"""

    if configured in {"min", "max"}:
        return configured
    return "min" if "loss" in monitor else "max"


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    topk: Sequence[int],
    binary_threshold: float = 0.5,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    logger: Any = None,
) -> dict[str, Any]:
    """在给定 DataLoader 上评估模型，返回完整指标字典。"""

    model.eval()
    meter = MetricAccumulator(
        num_classes=num_classes,
        topk=tuple(topk),
        binary_threshold=binary_threshold,
    )
    total_batches = len(loader)
    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled and device.type == "cuda",
        ):
            logits = model(inputs)
            loss = criterion(logits, _prepare_targets(targets, num_classes))
        meter.update(logits.detach().float(), targets, float(loss.item()))
        if logger is not None and total_batches and batch_index % max(1, total_batches // 5) == 0:
            logger.info("评估进度：%d/%d", batch_index, total_batches)
    return meter.compute()


def save_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    """把逐 epoch 指标写入 CSV；使用 BOM 兼容 Excel 中文环境。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_history(history: list[dict[str, Any]], path: Path) -> None:
    """绘制损失、准确率、F1 与学习率曲线。"""

    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, [row.get("train_loss") for row in history], label="train")
    axes[0, 0].plot(epochs, [row.get("val_loss") for row in history], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs, [row.get("train_accuracy") for row in history], label="train")
    axes[0, 1].plot(epochs, [row.get("val_accuracy") for row in history], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(epochs, [row.get("train_f1_macro") for row in history], label="train")
    axes[1, 0].plot(epochs, [row.get("val_f1_macro") for row in history], label="val")
    axes[1, 0].set_title("Macro F1")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    learning_rates = [float(row.get("lr") or 0.0) for row in history]
    axes[1, 1].plot(epochs, learning_rates, label="learning rate")
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].set_xlabel("Epoch")
    if learning_rates and all(value > 0 for value in learning_rates):
        axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_evaluation_artifacts(
    metrics: Mapping[str, Any],
    class_names: Sequence[str],
    output_dir: Path,
    prefix: str,
) -> None:
    """保存指标 JSON、逐类 CSV、混淆矩阵 CSV 与 PNG。"""

    save_json(dict(metrics), output_dir / f"{prefix}_metrics.json")
    save_per_class_csv(metrics, class_names, output_dir / f"{prefix}_per_class.csv")
    if "confusion_matrix" in metrics:
        save_confusion_csv(metrics["confusion_matrix"], class_names, output_dir / f"{prefix}_confusion.csv")
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            output_dir / f"{prefix}_confusion.png",
        )


class Trainer:
    """封装完整训练生命周期的主类。"""

    def __init__(
        self,
        config: dict[str, Any],
        model: nn.Module,
        bundle: DataBundle,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader | None,
        device: torch.device,
        run_dir: Path,
        logger: Any,
    ) -> None:
        self.config = config
        self.model = model
        self.bundle = bundle
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.run_dir = run_dir
        self.logger = logger

        self.train_cfg = config["train"]
        self.runtime_cfg = config["runtime"]
        self.checkpoint_cfg = config["checkpoint"]
        self.logging_cfg = config["logging"]
        self.evaluation_cfg = config["evaluation"]
        self.num_classes = bundle.num_classes
        self.class_names = bundle.class_names
        self.topk = tuple(int(k) for k in self.evaluation_cfg.get("topk", [1, 5]))
        self.binary_threshold = float(self.evaluation_cfg.get("binary_threshold", 0.5))

        # AMP 只对 CUDA 启用；CPU 上开启 fp16 autocast 容易变慢且数值不稳定。
        amp_cfg = self.train_cfg.get("amp") or {}
        self.amp_dtype = _amp_dtype(amp_cfg.get("dtype", "bfloat16"))
        self.amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
        if bool(amp_cfg.get("enabled", False)) and device.type != "cuda":
            logger.warning("当前设备不是 CUDA，已自动关闭混合精度训练。")

        self.grad_accumulation_steps = int(self.train_cfg.get("grad_accumulation_steps", 1))
        if self.grad_accumulation_steps < 1:
            raise ValueError("train.grad_accumulation_steps 必须 >= 1。")

        self.grad_clip_cfg = (self.train_cfg.get("backward") or {}).get("grad_clip") or {}
        self.stop_on_nan = bool(self.train_cfg.get("stop_on_nan", True))
        self.log_grad_norm = bool(self.train_cfg.get("log_grad_norm", False))
        self.print_freq = max(1, int(self.train_cfg.get("print_freq", 10)))

        # 类别权重可能来自训练标签，因此先构建数据再构建损失。
        train_targets = dataset_targets(bundle.train)
        self.criterion, self.loss_info = build_loss(
            config["loss"],
            self.num_classes,
            train_targets,
            device,
        )
        if self.num_classes == 1 and str(config["loss"].get("name")).lower() != "bce_with_logits":
            raise ValueError("num_classes=1 时请使用 bce_with_logits 损失。")

        self.optimizer = build_optimizer(model, config["optimizer"])
        self.scheduler = build_scheduler(
            self.optimizer,
            config["scheduler"],
            int(self.train_cfg["epochs"]),
            max(1, len(train_loader)),
        )
        self.scheduler_step_on = str(config["scheduler"].get("step_on", "epoch")).lower()
        if self.scheduler_step_on not in {"epoch", "batch"}:
            raise ValueError("scheduler.step_on 可选 epoch/batch。")

        amp_state = amp_cfg.get("grad_scaler") or {}
        self.scaler = _build_grad_scaler(
            enabled=self.amp_enabled and self.amp_dtype == torch.float16,
            init_scale=float(amp_state.get("init_scale", 65536.0)),
            growth_factor=float(amp_state.get("growth_factor", 2.0)),
            backoff_factor=float(amp_state.get("backoff_factor", 0.5)),
            growth_interval=int(amp_state.get("growth_interval", 2000)),
        )

        self.monitor = str(self.train_cfg.get("monitor", "val_loss"))
        self.monitor_mode = _monitor_mode(
            self.monitor,
            str(self.train_cfg.get("monitor_mode", "auto")),
        )
        self.best_metric = math.inf if self.monitor_mode == "min" else -math.inf
        self.best_epoch = 0
        self.early_counter = 0
        self.history: list[dict[str, Any]] = []
        self.start_epoch = 1
        self.total_seconds = 0.0

        self.best_path = run_dir / str(self.checkpoint_cfg.get("best_name", "best.pth"))
        self.last_path = run_dir / str(self.checkpoint_cfg.get("last_name", "last.pth"))
        self.history_path = run_dir / "history.csv"
        self.curves_path = run_dir / "curves.png"

        # 记录类别映射和损失配置，方便部署/复现。
        save_json(
            {
                "class_names": self.class_names,
                "num_classes": self.num_classes,
                "loss": self.loss_info,
                "input_shape": list(bundle.input_shape),
                "mean": list(bundle.mean),
                "std": list(bundle.std),
            },
            run_dir / "dataset_meta.json",
        )

        self._maybe_resume()
        self._maybe_compile()

    def _maybe_compile(self) -> None:
        """按配置尝试 torch.compile；失败时保留普通模型继续训练。"""

        compile_cfg = self.runtime_cfg.get("compile") or {}
        if not bool(compile_cfg.get("enabled", False)):
            return
        try:
            self.model = torch.compile(
                self.model,
                mode=str(compile_cfg.get("mode", "default")),
                fullgraph=bool(compile_cfg.get("fullgraph", False)),
                dynamic=bool(compile_cfg.get("dynamic", False)),
                backend=str(compile_cfg.get("backend", "inductor")),
            )
            self.logger.info("已启用 torch.compile（首次运行可能需要较长时间编译）。")
        except Exception as exc:  # Windows 或无 Triton 环境可能不支持
            self.logger.warning("torch.compile 启用失败，将使用普通 eager 模式：%s", exc)

    def _maybe_resume(self) -> None:
        """加载断点；模型、优化器、调度器、AMP scaler 与历史一并恢复。"""

        resume_value = self.checkpoint_cfg.get("resume_from")
        if resume_value in {None, "", False}:
            return
        resume_path = resolve_path(str(resume_value))
        if resume_path is None or not resume_path.exists():
            raise FileNotFoundError(f"checkpoint.resume_from 指向的断点不存在：{resume_path}")

        checkpoint = _load_checkpoint(resume_path, self.device)
        model_to_load = _unwrap_model(self.model)
        try:
            model_to_load.load_state_dict(checkpoint["model_state"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "断点中的模型结构与当前配置不一致，无法严格加载。"
                "请恢复原来的 model.layers/model.head 配置后再续训。\n"
                f"原始错误：{exc}"
            ) from exc

        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except (ValueError, KeyError) as exc:
                self.logger.warning("优化器状态恢复失败，将以新优化器状态继续：%s", exc)
        if self.scheduler is not None and checkpoint.get("scheduler_state"):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            except (ValueError, KeyError) as exc:
                self.logger.warning("学习率调度器状态恢复失败：%s", exc)
        if self.scaler is not None and checkpoint.get("scaler_state"):
            try:
                self.scaler.load_state_dict(checkpoint["scaler_state"])
            except (ValueError, KeyError) as exc:
                self.logger.warning("GradScaler 状态恢复失败：%s", exc)

        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_metric = float(checkpoint.get("best_metric", self.best_metric))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.history = list(checkpoint.get("history", []) or [])
        self.logger.info(
            "已从 %s 恢复，上一轮 epoch=%s，下一轮从 epoch=%s 开始。",
            resume_path,
            int(checkpoint.get("epoch", 0)),
            self.start_epoch,
        )

    def _autocast(self):
        """返回当前训练配置对应的 autocast 上下文。"""

        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _grad_norm(self) -> float:
        """计算全局梯度范数，支持 norm_type=inf 等 PyTorch 语义。"""

        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not parameters:
            return 0.0
        norm_type = float(self.grad_clip_cfg.get("norm_type", 2.0))
        norms = torch.stack([parameter.grad.detach().norm(p=norm_type) for parameter in parameters])
        total = torch.linalg.vector_norm(norms, ord=norm_type).item()
        return float(total)

    def _clip_gradients(self) -> float:
        """执行梯度裁剪，并返回裁剪前的全局范数。"""

        if not bool(self.grad_clip_cfg.get("enabled", False)):
            return self._grad_norm() if self.log_grad_norm else 0.0
        mode = str(self.grad_clip_cfg.get("mode", "norm")).lower()
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not parameters:
            return 0.0
        if mode == "norm":
            norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(self.grad_clip_cfg.get("max_norm", 1.0)),
                norm_type=float(self.grad_clip_cfg.get("norm_type", 2.0)),
            )
            return float(norm.item())
        if mode == "value":
            torch.nn.utils.clip_grad_value_(
                parameters,
                clip_value=float(self.grad_clip_cfg.get("value", 0.1)),
            )
            return self._grad_norm()
        if mode == "none":
            return self._grad_norm() if self.log_grad_norm else 0.0
        raise ValueError("train.backward.grad_clip.mode 可选 norm/value/none。")

    def _batch_accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """快速计算 batch 准确率，仅用于训练过程打印。"""

        with torch.no_grad():
            if self.num_classes == 1:
                prediction = (torch.sigmoid(logits[:, 0]) >= self.binary_threshold).long()
            else:
                prediction = logits.argmax(dim=1)
            return float((prediction == targets).float().mean().item())

    def _train_epoch(self) -> tuple[dict[str, Any], float, int, int]:
        """训练一个 epoch，返回指标、平均梯度范数、优化步数与跳过批次数。"""

        self.model.train()
        meter = MetricAccumulator(
            num_classes=self.num_classes,
            topk=self.topk,
            binary_threshold=self.binary_threshold,
        )
        self.optimizer.zero_grad(set_to_none=True)
        grad_norm_sum = 0.0
        grad_norm_count = 0
        optimizer_steps = 0
        skipped_batches = 0
        total_batches = len(self.train_loader)
        if total_batches == 0:
            raise ValueError("训练 DataLoader 没有 batch，请检查 batch_size/drop_last 配置。")
        epoch_start = time.perf_counter()

        for batch_index, (inputs, targets) in enumerate(self.train_loader, start=1):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with self._autocast():
                logits = self.model(inputs)
                loss = self.criterion(logits, _prepare_targets(targets, self.num_classes))

            # 非有限损失通常意味着学习率过大、数据含 NaN 或数值溢出。
            if not torch.isfinite(loss):
                skipped_batches += 1
                self.optimizer.zero_grad(set_to_none=True)
                message = f"Epoch {self.current_epoch} batch {batch_index}/{total_batches} 出现非有限损失。"
                if self.stop_on_nan:
                    raise RuntimeError(message + " 请降低学习率或检查数据。")
                self.logger.warning(message + " 已跳过该 batch。")
                continue

            # 梯度累积：把损失除以累积步数，使等效 batch 大小下的梯度尺度一致。
            backward_loss = loss / self.grad_accumulation_steps
            if self.scaler is not None:
                self.scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            do_step = (
                batch_index % self.grad_accumulation_steps == 0
                or batch_index == total_batches
            )
            if do_step:
                # AMP 的梯度只有在 unscale 后才能正确裁剪和计算范数。
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                norm = self._clip_gradients()
                if norm:
                    grad_norm_sum += norm
                    grad_norm_count += 1

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

                if self.scheduler is not None and self.scheduler_step_on == "batch":
                    self.scheduler.step()

            meter.update(logits.detach().float(), targets, float(loss.item()))

            if batch_index % self.print_freq == 0 or batch_index == total_batches:
                batch_accuracy = self._batch_accuracy(logits.detach(), targets)
                elapsed = time.perf_counter() - epoch_start
                self.logger.info(
                    "Epoch %d/%d | batch %d/%d | loss %.4f | acc %.2f%% | lr %.3e | %s",
                    self.current_epoch,
                    self.epochs,
                    batch_index,
                    total_batches,
                    float(loss.item()),
                    batch_accuracy * 100,
                    self._current_lr(),
                    format_seconds(elapsed),
                )

        metrics = meter.compute()
        average_norm = grad_norm_sum / grad_norm_count if grad_norm_count else 0.0
        return metrics, average_norm, optimizer_steps, skipped_batches

    def _evaluate(self, loader: torch.utils.data.DataLoader | None) -> dict[str, Any]:
        if loader is None:
            return {}
        return evaluate_loader(
            model=self.model,
            loader=loader,
            criterion=self.criterion,
            device=self.device,
            num_classes=self.num_classes,
            topk=self.topk,
            binary_threshold=self.binary_threshold,
            amp_enabled=self.amp_enabled,
            amp_dtype=self.amp_dtype,
            logger=self.logger,
        )

    def _monitor_value(self, train_metrics: Mapping[str, Any], val_metrics: Mapping[str, Any]) -> float:
        """从指标字典读取监控值，缺失时给出明确错误。"""

        mapping = {
            "train_loss": train_metrics.get("loss"),
            "train_accuracy": train_metrics.get("accuracy"),
            "val_loss": val_metrics.get("loss"),
            "val_accuracy": val_metrics.get("accuracy"),
            "val_f1_macro": val_metrics.get("f1_macro"),
        }
        if self.monitor not in mapping or mapping[self.monitor] is None:
            raise ValueError(f"监控指标 '{self.monitor}' 当前不可用。")
        return float(mapping[self.monitor])

    def _is_improvement(self, value: float) -> bool:
        min_delta = float(self.train_cfg.get("min_delta", 0.0))
        if self.monitor_mode == "min":
            return value < self.best_metric - min_delta
        return value > self.best_metric + min_delta

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Mapping[str, Any]) -> None:
        """保存完整训练状态；所有张量通过 state_dict 保存，避免直接 pickle 模型。"""

        model_to_save = _unwrap_model(self.model)
        payload = {
            "format_version": 1,
            "epoch": int(epoch),
            "model_state": model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "best_metric": float(self.best_metric),
            "best_epoch": int(self.best_epoch),
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "config": {
                key: value
                for key, value in self.config.items()
                if not str(key).startswith("_")
            },
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "input_shape": list(self.bundle.input_shape),
            "mean": list(self.bundle.mean),
            "std": list(self.bundle.std),
            "history": self.history,
            "metrics": dict(metrics),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再替换，避免训练中途断电留下半个检查点。
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _write_history(self) -> None:
        save_history_csv(self.history, self.history_path)
        save_json(self.history, self.run_dir / "history.json")

    def fit(self) -> dict[str, Any]:
        """执行完整训练流程，并返回最终结果摘要。"""

        self.epochs = int(self.train_cfg["epochs"])
        self.current_epoch = 0
        training_start = time.perf_counter()
        early_cfg = self.train_cfg.get("early_stopping") or {}
        early_enabled = bool(early_cfg.get("enabled", False))
        patience = int(early_cfg.get("patience", 5))

        self.logger.info(
            "开始训练：epochs=%d，batch_size=%d，训练样本=%d，验证样本=%d，可训练参数=%s",
            self.epochs,
            int(self.config["dataset"]["batch_size"]),
            self.bundle.train_size,
            self.bundle.val_size,
            sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad),
        )

        if self.start_epoch > self.epochs:
            self.logger.info("断点已完成全部 epoch，直接进入评估阶段。")

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.perf_counter()
            train_metrics, grad_norm, optimizer_steps, skipped = self._train_epoch()
            val_metrics = self._evaluate(self.val_loader)
            monitor_value = self._monitor_value(train_metrics, val_metrics)

            # 调度器按 epoch 更新；plateau 需要传入被监控指标。
            if self.scheduler is not None and self.scheduler_step_on == "epoch":
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()

            improved = self._is_improvement(monitor_value)
            if improved:
                self.best_metric = monitor_value
                self.best_epoch = epoch
                self.early_counter = 0
            elif early_enabled:
                self.early_counter += 1

            epoch_seconds = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "lr": self._current_lr(),
                "train_loss": train_metrics.get("loss", 0.0),
                "train_accuracy": train_metrics.get("accuracy", 0.0),
                "train_f1_macro": train_metrics.get("f1_macro", 0.0),
                "val_loss": val_metrics.get("loss", 0.0),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
                "val_f1_macro": val_metrics.get("f1_macro", 0.0),
                "best_metric": self.best_metric,
                "grad_norm": grad_norm,
                "optimizer_steps": optimizer_steps,
                "skipped_batches": skipped,
                "epoch_seconds": epoch_seconds,
            }
            self.history.append(row)
            self._write_history()

            self.logger.info(
                "Epoch %d/%d 完成 | train_loss %.4f | train_acc %.2f%% | "
                "val_loss %.4f | val_acc %.2f%% | val_f1 %.2f%% | lr %.3e | %s",
                epoch,
                self.epochs,
                train_metrics.get("loss", 0.0),
                train_metrics.get("accuracy", 0.0) * 100,
                val_metrics.get("loss", 0.0),
                val_metrics.get("accuracy", 0.0) * 100,
                val_metrics.get("f1_macro", 0.0) * 100,
                self._current_lr(),
                format_seconds(epoch_seconds),
            )

            # 保存顺序：先 best 再 last，确保 last 文件总是最新状态。
            if improved and bool(self.checkpoint_cfg.get("save_best", True)):
                self._save_checkpoint(self.best_path, epoch, val_metrics)
                self.logger.info("发现更优 %s=%.6f，已保存 %s", self.monitor, monitor_value, self.best_path.name)

            if bool(self.checkpoint_cfg.get("save_last", True)):
                self._save_checkpoint(self.last_path, epoch, val_metrics)

            every_n = int(self.checkpoint_cfg.get("save_every_n_epochs", 0))
            if every_n > 0 and epoch % every_n == 0:
                periodic_path = self.run_dir / f"epoch_{epoch:03d}.pth"
                self._save_checkpoint(periodic_path, epoch, val_metrics)
                self.logger.info("已保存周期检查点 %s", periodic_path.name)

            if early_enabled and self.early_counter >= patience:
                self.logger.info(
                    "早停触发：%s 已连续 %d 个 epoch 未提升（patience=%d）。",
                    self.monitor,
                    self.early_counter,
                    patience,
                )
                break

        self.total_seconds = time.perf_counter() - training_start
        if bool(self.logging_cfg.get("plot_curves", True)):
            plot_history(self.history, self.curves_path)
            self.logger.info("训练曲线已保存：%s", self.curves_path)

        # 训练结束后恢复最佳权重，避免最后几轮过拟合影响最终评估。
        restore_best = bool((self.train_cfg.get("early_stopping") or {}).get("restore_best_weights", True))
        if restore_best and self.best_path.exists():
            checkpoint = _load_checkpoint(self.best_path, self.device)
            _unwrap_model(self.model).load_state_dict(checkpoint["model_state"], strict=True)
            self.logger.info("已恢复监控指标最佳权重（epoch=%d, %s=%.6f）。", self.best_epoch, self.monitor, self.best_metric)

        test_metrics: dict[str, Any] = {}
        if bool(self.train_cfg.get("evaluate_test_after_train", True)) and self.test_loader is not None:
            self.logger.info("开始测试集评估……")
            test_metrics = self._evaluate(self.test_loader)
            save_evaluation_artifacts(test_metrics, self.class_names, self.run_dir, "test")
            self.logger.info(
                "测试集 | loss %.4f | acc %.2f%% | top%d %.2f%% | macro_f1 %.2f%%",
                test_metrics.get("loss", 0.0),
                test_metrics.get("accuracy", 0.0) * 100,
                min(self.topk) if self.topk else 1,
                test_metrics.get(f"top{min(self.topk) if self.topk else 1}_accuracy", 0.0) * 100,
                test_metrics.get("f1_macro", 0.0) * 100,
            )

        result = {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor": self.monitor,
            "monitor_mode": self.monitor_mode,
            "epochs_completed": self.history[-1]["epoch"] if self.history else 0,
            "training_seconds": self.total_seconds,
            "best_checkpoint": str(self.best_path) if self.best_path.exists() else None,
            "last_checkpoint": str(self.last_path) if self.last_path.exists() else None,
            "test_metrics": test_metrics,
        }
        save_json(result, self.run_dir / "train_result.json")
        return result

