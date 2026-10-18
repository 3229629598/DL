"""分类损失函数工厂与类别权重处理。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


def _balanced_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    """根据训练集类别频率计算逆频率权重。

    公式：weight[c] = N / (C * count[c])
    出现次数越少的类别权重越大。随后归一化到均值为 1，避免整体损失尺度
    随数据集大小发生明显变化。没有样本的类别权重设为 0。
    """

    counts = torch.bincount(torch.as_tensor(list(targets), dtype=torch.long), minlength=num_classes)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = len(targets) / (num_classes * counts[present].float())
    if present.any():
        weights[present] /= weights[present].mean()
    return weights


def _parse_weight(
    value: Any,
    targets: Sequence[int],
    num_classes: int,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    """把配置中的类别权重解析成张量。

    支持：
    - null：不加权。
    - balanced：按训练集频率自动平衡。
    - 长度为 num_classes 的数字列表：手工指定。
    """

    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() == "balanced":
            return _balanced_weights(targets, num_classes).to(device)
        raise ValueError(f"{name} 的字符串权重只支持 'balanced'。")
    if isinstance(value, (list, tuple)):
        if len(value) != num_classes:
            raise ValueError(f"{name} 的权重长度必须为 {num_classes}，当前为 {len(value)}。")
        return torch.tensor([float(item) for item in value], dtype=torch.float32, device=device)
    raise ValueError(f"{name} 必须是 null、balanced 或数字列表。")


def build_loss(
    loss_cfg: Mapping[str, Any],
    num_classes: int,
    train_targets: Sequence[int],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """创建分类损失函数，并返回用于日志记录的额外信息。"""

    name = str(loss_cfg.get("name", "cross_entropy")).lower()
    params = dict(loss_cfg.get("params") or {})
    info: dict[str, Any] = {"name": name}
    reduction = str(params.get("reduction", "mean")).lower()
    if reduction not in {"mean", "sum"}:
        raise ValueError(
            f"{name} 的 reduction 只支持 mean/sum；none 会返回逐样本损失，"
            "无法直接驱动本项目的单值训练循环。"
        )

    if name == "cross_entropy":
        if num_classes < 2:
            raise ValueError("cross_entropy 需要至少 2 个类别；二分类单输出请用 bce_with_logits。")
        weight_value = params.pop("weight", None)
        weight = _parse_weight(
            weight_value, train_targets, num_classes, device, "loss.params.weight"
        )
        criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=int(params.get("ignore_index", -100)),
            reduction=reduction,
            label_smoothing=float(params.get("label_smoothing", 0.0)),
        )
        info["weight"] = weight.detach().cpu().tolist() if weight is not None else None
        info["label_smoothing"] = float(params.get("label_smoothing", 0.0))
        return criterion, info

    if name == "bce_with_logits":
        if num_classes != 1:
            raise ValueError(
                "bce_with_logits 适用于 num_classes=1 的单 logit 二分类；"
                "多分类请使用 cross_entropy。"
            )
        # BCE 的 pos_weight 只针对正类，因此用负/正样本比例。
        counts = torch.bincount(
            torch.as_tensor(list(train_targets), dtype=torch.long),
            minlength=2,
        )
        positive = int(counts[1].item())
        negative = int(counts[0].item())
        pos_weight_value = params.pop("pos_weight", None)
        if isinstance(pos_weight_value, str) and pos_weight_value.lower() == "balanced":
            pos_weight = torch.tensor(
                [negative / max(positive, 1)],
                dtype=torch.float32,
                device=device,
            )
        elif pos_weight_value is None:
            pos_weight = None
        else:
            pos_weight = torch.tensor(
                [float(pos_weight_value)],
                dtype=torch.float32,
                device=device,
            )
        weight = _parse_weight(
            params.pop("weight", None), train_targets, 1, device, "loss.params.weight"
        )
        criterion = nn.BCEWithLogitsLoss(
            weight=weight,
            pos_weight=pos_weight,
            reduction=reduction,
        )
        info["pos_weight"] = float(pos_weight.item()) if pos_weight is not None else None
        return criterion, info

    if name == "nll":
        if num_classes < 2:
            raise ValueError("nll 需要至少 2 个类别，并且模型输出必须已做 log_softmax。")
        weight = _parse_weight(
            params.pop("weight", None), train_targets, num_classes, device, "loss.params.weight"
        )
        criterion = nn.NLLLoss(
            weight=weight,
            ignore_index=int(params.get("ignore_index", -100)),
            reduction=reduction,
        )
        info["weight"] = weight.detach().cpu().tolist() if weight is not None else None
        return criterion, info

    raise ValueError(f"未知损失函数 '{name}'，可选 cross_entropy/bce_with_logits/nll。")

