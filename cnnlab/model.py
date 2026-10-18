"""把配置中的 layers/head 列表组装成完整 CNN，并输出结构摘要。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .layers import BuildContext, build_layers
from .utils import count_parameters, format_number


class ConfigurableCNN(nn.Module):
    """由配置文件动态生成的特征提取器 + 分类头。

    features 对应 model.layers，head 对应 model.head。
    两部分都只是标准 nn.Sequential，因此 PyTorch 会自动构建计算图并完成反向传播。
    """

    def __init__(
        self,
        feature_blocks: list[nn.Module],
        head_blocks: list[nn.Module],
        input_shape: tuple[int, ...],
        feature_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        num_classes: int,
        channels_last: bool = False,
        build_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(*feature_blocks)
        self.head = nn.Sequential(*head_blocks)
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.feature_shape = tuple(int(dim) for dim in feature_shape)
        self.output_shape = tuple(int(dim) for dim in output_shape)
        self.num_classes = int(num_classes)
        self.channels_last = bool(channels_last)
        self.build_metadata = dict(build_metadata or {})

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """前向传播。

        return_features=True 时返回 (logits, features)，可用于特征可视化、
        迁移学习或自定义损失；默认只返回 logits。
        """

        # channels_last 对支持该内存格式的卷积通常更高效；输入需为 4 维。
        if self.channels_last and x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        features = self.features(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """只运行特征提取部分，不经过分类头。"""

        if self.channels_last and x.dim() == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        return self.features(x)

    def architecture_summary(self) -> str:
        """生成包含每层输出形状与参数量的中文结构表。"""

        lines = [
            "=" * 96,
            "网络结构概览",
            "=" * 96,
            f"输入形状（不含 batch）：{self.input_shape}",
            f"特征形状（不含 batch）：{self.feature_shape}",
            f"输出形状（不含 batch）：{self.output_shape}",
            f"类别数：{self.num_classes}",
            "-" * 96,
            f"{'层名称':<28}{'类型':<20}{'输出形状':<24}{'参数量':>12}  {'可训练':<6}",
            "-" * 96,
        ]

        def append_metadata(items: list[dict[str, Any]], indent: int = 0) -> None:
            for item in items:
                name = "  " * indent + str(item.get("name", ""))
                lines.append(
                    f"{name:<28}{str(item.get('type', '')):<20}"
                    f"{str(item.get('shape', '')):<24}"
                    f"{format_number(int(item.get('parameters', 0))):>12}  "
                    f"{'是' if item.get('trainable', True) else '否':<6}"
                )
                children = item.get("children") or []
                if children:
                    append_metadata(children, indent + 1)

        append_metadata(self.build_metadata.get("features", []))
        append_metadata(self.build_metadata.get("head", []))
        total = count_parameters(self)
        trainable = count_parameters(self, trainable_only=True)
        lines.extend(
            [
                "-" * 96,
                f"总参数量：{total:,} ({format_number(total)})",
                f"可训练参数量：{trainable:,} ({format_number(trainable)})",
                "=" * 96,
            ]
        )
        return "\n".join(lines)


def build_model(
    config: Mapping[str, Any],
    input_shape: tuple[int, int, int],
    num_classes: int,
) -> ConfigurableCNN:
    """根据配置构建网络。

    input_shape 来自数据集（[C,H,W]），也可由 model.input_shape 显式覆盖。
    构建过程中会逐层推导形状，因此大多数通道数、特征数都不需要手工填写。
    """

    model_cfg = config["model"]
    configured_shape = model_cfg.get("input_shape")
    if configured_shape is not None:
        configured = tuple(int(dim) for dim in configured_shape)
        if configured != tuple(input_shape):
            raise ValueError(
                f"model.input_shape={configured} 与数据集推断的输入形状 {tuple(input_shape)} 不一致。"
                "请修改配置或删除 model.input_shape 让程序自动使用数据形状。"
            )

    defaults = dict(model_cfg.get("defaults") or {})
    context = BuildContext(num_classes=int(num_classes), defaults=defaults)

    feature_blocks, feature_shape, feature_meta = build_layers(
        model_cfg["layers"],
        tuple(int(dim) for dim in input_shape),
        context,
        prefix="features",
    )
    head_blocks, output_shape, head_meta = build_layers(
        model_cfg.get("head", []),
        feature_shape,
        context,
        prefix="head",
    )

    expected = (int(num_classes),)
    if output_shape != expected:
        binary_ok = int(num_classes) == 1 and output_shape == (1,)
        if not binary_ok:
            raise ValueError(
                f"网络最终输出形状为 {output_shape}，但期望 {expected}。"
                "请确认 model.head 最后是 out_features=num_classes 的 linear 层，"
                "并在其前面加入 flatten 或 global_avgpool。"
            )

    model = ConfigurableCNN(
        feature_blocks=feature_blocks,
        head_blocks=head_blocks,
        input_shape=tuple(int(dim) for dim in input_shape),
        feature_shape=feature_shape,
        output_shape=output_shape,
        num_classes=int(num_classes),
        channels_last=bool(model_cfg.get("channels_last", False)),
        build_metadata={"features": feature_meta, "head": head_meta},
    )
    return model

