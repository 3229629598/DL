"""可配置网络层的工厂与基础模块。

核心思想：
- 每个配置项描述一个“层块”（LayerBlock）。
- 层块由核心运算（卷积/池化/线性/残差等）以及可选的归一化、激活、Dropout 组成。
- 每个层块自己记录输出形状，构建器据此自动推导下一层输入维度。
- 所有模块都是标准 nn.Module，因此 PyTorch 的自动求导（反向传播）无需手写；
  反向传播的可调部分（损失、优化器、梯度裁剪、AMP、梯度累积、梯度检查点）
  由训练配置控制。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# 常见别名统一映射到内部名称，让配置写起来更自然。
TYPE_ALIASES = {
    "conv": "conv2d",
    "convolution": "conv2d",
    "depthwise_conv": "conv2d",
    "dense": "linear",
    "fc": "linear",
    "max_pool": "maxpool2d",
    "avg_pool": "avgpool2d",
    "global_pool": "global_avgpool",
    "bn": "batchnorm2d",
    "batchnorm": "batchnorm2d",
    "instance_norm": "instancenorm2d",
    "layer_norm": "layernorm",
    "group_norm": "groupnorm",
    "drop_out": "dropout",
}


def _normalize_type(value: Any) -> str:
    """把层类型规范成小写、下划线形式，并展开常见别名。"""

    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return TYPE_ALIASES.get(name, name)


def _as_pair(value: Any, name: str) -> tuple[int, int]:
    """把整数或长度为 1/2 的列表转换成 (高, 宽) 二元组。"""

    if isinstance(value, int) and not isinstance(value, bool):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) in {1, 2}:
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    raise ValueError(f"{name} 必须是整数或长度为 1/2 的列表，当前为 {value!r}。")


def _build_activation(spec: Any, layer_name: str) -> nn.Module:
    """根据字符串或字典创建激活函数。

    字符串形式：activation: relu
    字典形式：activation: {name: leaky_relu, negative_slope: 0.1}
    设为 null / none 时使用 Identity，便于在配置中显式关闭激活。
    """

    if spec is None:
        return nn.Identity()
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, Mapping):
        params = dict(spec)
        name = params.pop("name", params.pop("type", None))
        if name is None:
            raise ValueError(f"层 {layer_name} 的激活函数字典缺少 name 字段。")
    else:
        raise ValueError(f"层 {layer_name} 的 activation 必须是字符串、字典或 null。")

    name = str(name).strip().lower().replace("-", "_")
    if name in {"none", "identity", "linear"}:
        return nn.Identity()

    # 延迟导入 nn 中的类，通过类构造器统一处理各激活函数的专属参数。
    registry: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "celu": nn.CELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "hardsigmoid": nn.Hardsigmoid,
        "hardswish": nn.Hardswish,
        "softplus": nn.Softplus,
        "softsign": nn.Softsign,
        "prelu": nn.PReLU,
        "logsigmoid": nn.LogSigmoid,
    }
    if name == "relu6":
        # PyTorch 自带 Hardtanh，可精确实现 ReLU6。
        return nn.Hardtanh(
            min_val=0.0,
            max_val=6.0,
            inplace=bool(params.get("inplace", False)),
        )
    if name in {"log_softmax", "softmax"}:
        # NLLLoss 需要 log_softmax 输出；这里同时提供 softmax 便于其他场景使用。
        cls = nn.LogSoftmax if name == "log_softmax" else nn.Softmax
        return cls(dim=int(params.get("dim", 1)))
    if name not in registry:
        raise ValueError(
            f"层 {layer_name} 的激活函数 '{name}' 不受支持。"
            f"可选：{', '.join(sorted(registry))}, relu6, softmax, log_softmax, none。"
        )

    try:
        return registry[name](**params)
    except TypeError as exc:
        raise ValueError(f"层 {layer_name} 的激活函数参数不正确：{exc}") from exc


def _build_norm(spec: Any, in_shape: tuple[int, ...], layer_name: str) -> nn.Module:
    """创建归一化层，并从输入形状自动推导通道数/特征数。"""

    if spec is None:
        return nn.Identity()
    if isinstance(spec, str):
        params: dict[str, Any] = {"type": spec}
    elif isinstance(spec, Mapping):
        params = dict(spec)
    else:
        raise ValueError(f"层 {layer_name} 的 norm 必须是字符串、字典或 null。")

    norm_type = _normalize_type(params.pop("type", params.pop("name", "")))
    if norm_type in {"", "none", "identity"}:
        return nn.Identity()

    if norm_type in {"batchnorm2d", "instancenorm2d"}:
        if len(in_shape) != 3:
            raise ValueError(
                f"层 {layer_name} 使用 {norm_type}，但输入形状为 {in_shape}；"
                "该归一化要求 4 维特征 [N,C,H,W]。"
            )
        channels = int(in_shape[0])
        params.pop("num_features", None)
        cls = nn.BatchNorm2d if norm_type == "batchnorm2d" else nn.InstanceNorm2d
        try:
            return cls(num_features=channels, **params)
        except TypeError as exc:
            raise ValueError(f"层 {layer_name} 的 {norm_type} 参数不正确：{exc}") from exc

    if norm_type == "batchnorm1d":
        if len(in_shape) != 1:
            raise ValueError(
                f"层 {layer_name} 使用 batchnorm1d，但输入形状为 {in_shape}；"
                "该归一化要求二维特征 [N,features]。"
            )
        params.pop("num_features", None)
        try:
            return nn.BatchNorm1d(num_features=int(in_shape[0]), **params)
        except TypeError as exc:
            raise ValueError(f"层 {layer_name} 的 batchnorm1d 参数不正确：{exc}") from exc

    if norm_type == "layernorm":
        # 默认归一化最后一维；也允许用 normalized_shape 指定多个尾随维度。
        normalized_shape = params.pop("normalized_shape", in_shape[-1])
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(int(dim) for dim in normalized_shape)
        try:
            return nn.LayerNorm(normalized_shape=normalized_shape, **params)
        except TypeError as exc:
            raise ValueError(f"层 {layer_name} 的 layernorm 参数不正确：{exc}") from exc

    if norm_type == "groupnorm":
        if len(in_shape) < 1:
            raise ValueError(f"层 {layer_name} 的输入形状不能为空。")
        num_groups = int(params.pop("num_groups", 1))
        channels = int(in_shape[0])
        if channels % num_groups != 0:
            raise ValueError(
                f"层 {layer_name} 的 groupnorm 要求通道数 {channels} 能被 num_groups={num_groups} 整除。"
            )
        params.pop("num_channels", None)
        try:
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels, **params)
        except TypeError as exc:
            raise ValueError(f"层 {layer_name} 的 groupnorm 参数不正确：{exc}") from exc

    raise ValueError(
        f"层 {layer_name} 的归一化类型 '{norm_type}' 不支持。"
        "可选：batchnorm2d, batchnorm1d, instancenorm2d, layernorm, groupnorm, none。"
    )


def _build_dropout(spec: Any, in_shape: tuple[int, ...], layer_name: str) -> nn.Module:
    """创建 Dropout；数字 0.2 等价于 {type: dropout, p: 0.2}。"""

    if spec is None:
        return nn.Identity()
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        params: dict[str, Any] = {"p": float(spec), "type": "dropout"}
    elif isinstance(spec, str):
        params = {"p": 0.5, "type": spec}
    elif isinstance(spec, Mapping):
        params = dict(spec)
    else:
        raise ValueError(f"层 {layer_name} 的 dropout 必须是数字、字符串、字典或 null。")

    dropout_type = _normalize_type(params.pop("type", "dropout"))
    probability = float(params.pop("p", 0.5))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"层 {layer_name} 的 dropout 概率必须在 [0,1]，当前为 {probability}。")
    if probability == 0.0 or dropout_type in {"none", "identity"}:
        return nn.Identity()

    if dropout_type in {"dropout", "drop"}:
        return nn.Dropout(p=probability, **params)
    if dropout_type == "dropout2d":
        if len(in_shape) != 3:
            raise ValueError(f"层 {layer_name} 使用 dropout2d，但输入不是 4 维特征：{in_shape}。")
        return nn.Dropout2d(p=probability, **params)
    if dropout_type == "alpha_dropout":
        return nn.AlphaDropout(p=probability, **params)
    raise ValueError(f"层 {layer_name} 的 dropout 类型 '{dropout_type}' 不支持。")


def _apply_weight_init(module: nn.Module, spec: Any, layer_name: str) -> None:
    """初始化卷积/线性权重，并重置归一化层的权重与偏置。

    只对传入模块及其普通子模块生效；复合层（sequence/residual）的内部层
    应各自配置 weight_init，避免外层配置意外覆盖内层精心设置的初始化。
    """

    if spec is None:
        return
    if isinstance(spec, str):
        params: dict[str, Any] = {"method": spec}
    elif isinstance(spec, Mapping):
        params = dict(spec)
    else:
        raise ValueError(f"层 {layer_name} 的 weight_init 必须是字符串、字典或 null。")

    method = str(params.pop("method", "default")).lower().replace("-", "_")
    if method in {"default", "none", "pytorch"}:
        return

    bias_value = float(params.pop("bias_value", 0.0))
    for child in module.modules():
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            weight = child.weight
            try:
                if method == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        weight,
                        a=float(params.get("a", 0.0)),
                        mode=str(params.get("mode", "fan_out")),
                        nonlinearity=str(params.get("nonlinearity", "relu")),
                    )
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        weight,
                        a=float(params.get("a", math.sqrt(5))),
                        mode=str(params.get("mode", "fan_in")),
                        nonlinearity=str(params.get("nonlinearity", "leaky_relu")),
                    )
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(weight, gain=float(params.get("gain", 1.0)))
                elif method == "xavier_uniform":
                    nn.init.xavier_uniform_(weight, gain=float(params.get("gain", 1.0)))
                elif method == "normal":
                    nn.init.normal_(
                        weight,
                        mean=float(params.get("mean", 0.0)),
                        std=float(params.get("std", 0.02)),
                    )
                elif method == "uniform":
                    bound = float(params.get("bound", 0.05))
                    nn.init.uniform_(weight, -bound, bound)
                elif method == "orthogonal":
                    nn.init.orthogonal_(weight, gain=float(params.get("gain", 1.0)))
                elif method == "dirac":
                    nn.init.dirac_(weight)
                elif method == "zeros":
                    nn.init.zeros_(weight)
                elif method == "ones":
                    nn.init.ones_(weight)
                else:
                    raise ValueError(
                        f"层 {layer_name} 的初始化方法 '{method}' 不支持。"
                        "可选 kaiming_normal/kaiming_uniform/xavier_normal/xavier_uniform/"
                        "normal/uniform/orthogonal/dirac/zeros/ones/default。"
                    )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"层 {layer_name} 的权重初始化失败：{exc}") from exc

            if child.bias is not None:
                nn.init.constant_(child.bias, bias_value)

        # 归一化层的默认初始化必须保留：权重为 1、偏置为 0。
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            if child.weight is not None and child.affine:
                nn.init.ones_(child.weight)
            if child.bias is not None and child.affine:
                nn.init.zeros_(child.bias)
        if isinstance(child, nn.LayerNorm):
            if child.weight is not None:
                nn.init.ones_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class Conv2dCore(nn.Module):
    """带“same”填充能力的二维卷积。

    PyTorch 原生 Conv2d 的 padding 是左右/上下对称的，无法完整表达 stride>1
    时的 same 填充。这里在 forward 中按输入尺寸动态计算左右/上下填充量，
    因此支持任意输入分辨率以及 stride、dilation 的组合。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        groups: int,
        bias: bool,
        padding_mode: str,
        same: bool,
        constant_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.same = same
        self.constant_value = float(constant_value)
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0) if same else padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros" if same else padding_mode,
        )

    @staticmethod
    def _same_padding(
        size: int,
        kernel: int,
        stride: int,
        dilation: int,
    ) -> tuple[int, int]:
        """计算某一维的总填充量，并尽量对称分配。

        输出尺寸与 PyTorch 的 ceil_mode 思路一致：ceil(size / stride)。
        返回 (前侧填充, 后侧填充)，差值最多为 1。
        """

        output_size = math.ceil(size / stride)
        total = max((output_size - 1) * stride + dilation * (kernel - 1) + 1 - size, 0)
        left = total // 2
        return left, total - left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.same:
            return self.conv(x)

        # F.pad 的顺序是 (左, 右, 上, 下)。
        top, bottom = self._same_padding(
            x.shape[2],
            self.kernel_size[0],
            self.stride[0],
            self.dilation[0],
        )
        left, right = self._same_padding(
            x.shape[3],
            self.kernel_size[1],
            self.stride[1],
            self.dilation[1],
        )

        mode_map = {
            "zeros": "constant",
            "reflect": "reflect",
            "replicate": "replicate",
            "circular": "circular",
        }
        if self.padding_mode not in mode_map:
            raise ValueError(f"不支持的 padding_mode：{self.padding_mode}")
        mode = mode_map[self.padding_mode]
        # 当填充量为 0 时直接卷积，避免 F.pad 对 reflect/circular 的限制。
        if top or bottom or left or right:
            if mode == "constant":
                x = F.pad(x, (left, right, top, bottom), mode=mode, value=self.constant_value)
            else:
                x = F.pad(x, (left, right, top, bottom), mode=mode)
        return self.conv(x)

    def extra_repr(self) -> str:
        return (
            f"same={self.same}, kernel={self.kernel_size}, stride={self.stride}, "
            f"dilation={self.dilation}, padding_mode={self.padding_mode}"
        )


class GlobalPool2d(nn.Module):
    """把 [N,C,H,W] 压成 [N,C]，用于分类头之前。"""

    def __init__(self, mode: str = "avg") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "avg":
            return x.mean(dim=(2, 3))
        if self.mode == "max":
            return x.amax(dim=(2, 3))
        raise ValueError(f"未知全局池化模式：{self.mode}")


class ResidualCore(nn.Module):
    """通用残差核心：y = F(x) + P(x)。

    F 是配置中的内部层序列；P 是投影分支。形状相同时 P 为恒等映射，
    形状不同时可自动使用 1x1 卷积或线性层投影。若空间尺寸不同，
    投影特征会用最近邻插值对齐到 F(x) 的尺寸。
    """

    def __init__(self, inner: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.projection = projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.inner(x)
        identity = self.projection(x)
        if identity.shape != output.shape and identity.dim() == 4:
            identity = F.interpolate(identity, size=output.shape[2:], mode="nearest")
        if identity.shape != output.shape:
            raise RuntimeError(
                f"残差分支形状 {tuple(identity.shape)} 与主分支 {tuple(output.shape)} 不一致，"
                "请检查 projection 与内部层配置。"
            )
        return output + identity


class LayerBlock(nn.Module):
    """核心运算 + 可选归一化/激活/Dropout 的组合层。

    order 决定执行顺序，默认：
    core -> norm -> activation -> dropout

    例如想使用“先激活再归一化”的结构，可配置：
    order: [core, activation, norm, dropout]
    """

    VALID_PARTS = {"core", "norm", "activation", "dropout"}

    def __init__(
        self,
        core: nn.Module,
        norm: nn.Module,
        activation: nn.Module,
        dropout: nn.Module,
        order: list[str] | tuple[str, ...],
        layer_name: str,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.core = core
        # 使用 ModuleDict 便于按 order 中的字符串动态调用，同时保证参数被正确注册。
        self.parts = nn.ModuleDict(
            {
                "norm": norm,
                "activation": activation,
                "dropout": dropout,
            }
        )
        self.order = [str(part).lower() for part in order]
        self.layer_name = layer_name
        self.use_checkpoint = bool(use_checkpoint)
        self._validate_order()

    def _validate_order(self) -> None:
        if self.order.count("core") != 1:
            raise ValueError(f"层 {self.layer_name} 的 order 必须且只能包含一次 core。")
        if len(set(self.order)) != len(self.order):
            raise ValueError(f"层 {self.layer_name} 的 order 中存在重复项：{self.order}。")
        unknown = set(self.order) - self.VALID_PARTS
        if unknown:
            raise ValueError(
                f"层 {self.layer_name} 的 order 包含未知项 {sorted(unknown)}；"
                "可选 core/norm/activation/dropout。"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for part in self.order:
            if part == "core":
                # 梯度检查点用“少存激活、反向时重算”换显存，适合大模型/小显存。
                if self.use_checkpoint and self.training and any(
                    parameter.requires_grad for parameter in self.core.parameters()
                ):
                    x = checkpoint(self.core, x, use_reentrant=False)
                else:
                    x = self.core(x)
            else:
                x = self.parts[part](x)
        return x


@dataclass
class BuildContext:
    """构建网络时共享的上下文信息。"""

    num_classes: int
    defaults: dict[str, Any]


def _parse_order(layer_cfg: Mapping[str, Any], defaults: Mapping[str, Any]) -> list[str]:
    """读取层执行顺序；既支持列表，也支持用箭头连接的字符串。"""

    raw = layer_cfg.get("order", defaults.get("order", ["core", "norm", "activation", "dropout"]))
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.replace("->", ",").split(",") if part.strip()]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"order 必须是列表或 'core->norm->activation' 形式的字符串，当前为 {raw!r}。")
    return [str(part) for part in raw]


def _conv_output_shape(
    height: int,
    width: int,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    """按照 PyTorch 卷积公式计算输出高宽。"""

    out_h = math.floor((height + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
    out_w = math.floor((width + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"卷积输出尺寸非法：输入 {(height, width)}，kernel={kernel}，stride={stride}，"
            f"padding={padding}，dilation={dilation}。请减小 kernel/stride 或增大 padding。"
        )
    return out_h, out_w


def _pool_output_shape(
    height: int,
    width: int,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    ceil_mode: bool,
) -> tuple[int, int]:
    """按照 PyTorch 池化规则计算输出高宽，包括 ceil_mode 的边界修正。"""

    def one_dim(size: int, k: int, s: int, p: int) -> int:
        if ceil_mode:
            out = math.ceil((size + 2 * p - k) / s) + 1
            # PyTorch 保证最后一个池化窗口必须从输入（含右/下 padding）内部开始。
            if (out - 1) * s >= size + p:
                out -= 1
        else:
            out = math.floor((size + 2 * p - k) / s) + 1
        return out

    out_h = one_dim(height, kernel[0], stride[0], padding[0])
    out_w = one_dim(width, kernel[1], stride[1], padding[1])
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"池化输出尺寸非法：输入 {(height, width)}，kernel={kernel}，stride={stride}，padding={padding}。"
        )
    return out_h, out_w


def _build_core(
    layer_type: str,
    layer_cfg: dict[str, Any],
    in_shape: tuple[int, ...],
    has_norm: bool,
    context: BuildContext,
    layer_name: str,
) -> tuple[nn.Module, tuple[int, ...], list[dict[str, Any]]]:
    """构建单个核心运算，返回 (模块, 输出形状, 子层摘要)。"""

    children: list[dict[str, Any]] = []

    if layer_type == "conv2d":
        if len(in_shape) != 3:
            raise ValueError(
                f"层 {layer_name} 是 conv2d，但输入形状为 {in_shape}；"
                "卷积要求 [C,H,W]，请检查前一层或 input_shape。"
            )
        depthwise = bool(layer_cfg.get("depthwise", False)) or _normalize_type(
            layer_cfg.get("mode", "")
        ) == "depthwise"
        in_channels = int(in_shape[0])
        out_channels_raw = layer_cfg.get("out_channels", layer_cfg.get("filters"))
        if out_channels_raw is None or (
            isinstance(out_channels_raw, str) and out_channels_raw.lower() == "same"
        ):
            if depthwise:
                out_channels = in_channels * int(layer_cfg.get("depth_multiplier", 1))
            else:
                raise ValueError(f"层 {layer_name} 的 conv2d 必须配置 out_channels（或写 same）。")
        else:
            out_channels = int(out_channels_raw)

        kernel_size = _as_pair(layer_cfg.get("kernel_size", 3), f"{layer_name}.kernel_size")
        stride = _as_pair(layer_cfg.get("stride", 1), f"{layer_name}.stride")
        dilation = _as_pair(layer_cfg.get("dilation", 1), f"{layer_name}.dilation")
        if any(value <= 0 for value in (*kernel_size, *stride, *dilation)):
            raise ValueError(f"层 {layer_name} 的 kernel_size/stride/dilation 必须为正数。")

        padding_raw = layer_cfg.get("padding", 0)
        same = isinstance(padding_raw, str) and padding_raw.lower() == "same"
        if same:
            padding = (0, 0)
        elif isinstance(padding_raw, str) and padding_raw.lower() == "valid":
            padding = (0, 0)
        else:
            padding = _as_pair(padding_raw, f"{layer_name}.padding")
        if any(value < 0 for value in padding):
            raise ValueError(f"层 {layer_name} 的 padding 不能为负数。")

        groups_raw = layer_cfg.get("groups", 1)
        if depthwise:
            groups = in_channels
        elif isinstance(groups_raw, str) and groups_raw.lower() == "depthwise":
            groups = in_channels
        elif isinstance(groups_raw, str) and groups_raw.lower() == "auto":
            groups = math.gcd(in_channels, out_channels)
        else:
            groups = int(groups_raw)
        if groups <= 0 or in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(
                f"层 {layer_name} 的分组卷积参数非法：in_channels={in_channels}, "
                f"out_channels={out_channels}, groups={groups}，两者都必须能被 groups 整除。"
            )

        padding_mode = str(layer_cfg.get("padding_mode", "zeros")).lower()
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError(
                f"层 {layer_name} 的 padding_mode='{padding_mode}' 不支持，"
                "可选 zeros/reflect/replicate/circular。"
            )
        bias_raw = layer_cfg.get("bias")
        bias = (not has_norm) if bias_raw is None else bool(bias_raw)

        core = Conv2dCore(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            same=same,
            constant_value=float(layer_cfg.get("padding_value", 0.0)),
        )
        if same:
            out_h = math.ceil(in_shape[1] / stride[0])
            out_w = math.ceil(in_shape[2] / stride[1])
        else:
            out_h, out_w = _conv_output_shape(
                int(in_shape[1]),
                int(in_shape[2]),
                kernel_size,
                stride,
                padding,
                dilation,
            )
        return core, (out_channels, out_h, out_w), children

    if layer_type == "linear":
        if len(in_shape) != 1:
            raise ValueError(
                f"层 {layer_name} 是 linear，但输入形状为 {in_shape}；"
                "请在 linear 前加入 flatten 或 global_avgpool 层。"
            )
        out_features_raw = layer_cfg.get("out_features", layer_cfg.get("units"))
        if out_features_raw in {"num_classes", "auto"}:
            out_features = context.num_classes
        elif out_features_raw is None:
            raise ValueError(f"层 {layer_name} 的 linear 必须配置 out_features。")
        else:
            out_features = int(out_features_raw)
        bias = bool(layer_cfg.get("bias", True))
        core = nn.Linear(int(in_shape[0]), out_features, bias=bias)
        return core, (out_features,), children

    if layer_type in {"maxpool2d", "avgpool2d"}:
        if len(in_shape) != 3:
            raise ValueError(f"层 {layer_name} 是 {layer_type}，但输入形状为 {in_shape}。")
        kernel = _as_pair(layer_cfg.get("kernel_size", 2), f"{layer_name}.kernel_size")
        stride_raw = layer_cfg.get("stride")
        stride = kernel if stride_raw is None else _as_pair(stride_raw, f"{layer_name}.stride")
        padding = _as_pair(layer_cfg.get("padding", 0), f"{layer_name}.padding")
        ceil_mode = bool(layer_cfg.get("ceil_mode", False))
        params = {
            "kernel_size": kernel,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
        }
        if layer_type == "avgpool2d":
            params["count_include_pad"] = bool(layer_cfg.get("count_include_pad", True))
            if layer_cfg.get("divisor_override") is not None:
                params["divisor_override"] = int(layer_cfg["divisor_override"])
            core: nn.Module = nn.AvgPool2d(**params)
        else:
            params["return_indices"] = False
            core = nn.MaxPool2d(**params)
        out_h, out_w = _pool_output_shape(
            int(in_shape[1]),
            int(in_shape[2]),
            kernel,
            stride,
            padding,
            ceil_mode,
        )
        return core, (int(in_shape[0]), out_h, out_w), children

    if layer_type in {"adaptive_avgpool2d", "adaptive_maxpool2d"}:
        if len(in_shape) != 3:
            raise ValueError(f"层 {layer_name} 是 {layer_type}，但输入形状为 {in_shape}。")
        output_size = _as_pair(layer_cfg.get("output_size", 1), f"{layer_name}.output_size")
        cls = nn.AdaptiveAvgPool2d if layer_type == "adaptive_avgpool2d" else nn.AdaptiveMaxPool2d
        return cls(output_size=output_size), (int(in_shape[0]), *output_size), children

    if layer_type in {"global_avgpool", "global_maxpool"}:
        if len(in_shape) != 3:
            raise ValueError(f"层 {layer_name} 是 {layer_type}，但输入形状为 {in_shape}。")
        mode = "avg" if layer_type == "global_avgpool" else "max"
        return GlobalPool2d(mode), (int(in_shape[0]),), children

    if layer_type == "flatten":
        start_dim = int(layer_cfg.get("start_dim", 1))
        if start_dim < 1:
            raise ValueError(f"层 {layer_name} 的 flatten.start_dim 必须 >= 1（0 是 batch 维）。")
        if start_dim > len(in_shape):
            raise ValueError(
                f"层 {layer_name} 的 flatten.start_dim={start_dim} 超出输入形状 {in_shape}。"
            )
        # in_shape 不含 batch，因此 start_dim=1 对应从 in_shape[0] 开始展平。
        flat_size = math.prod(in_shape[start_dim - 1 :])
        out_shape = (*in_shape[: start_dim - 1], flat_size)
        return nn.Flatten(start_dim=start_dim), tuple(out_shape), children

    if layer_type == "identity":
        return nn.Identity(), in_shape, children

    if layer_type == "sequence":
        inner_specs = layer_cfg.get("layers")
        if not isinstance(inner_specs, list) or not inner_specs:
            raise ValueError(f"sequence 层 {layer_name} 必须包含非空的 layers 列表。")
        blocks, out_shape, metas = build_layers(
            inner_specs,
            in_shape,
            context,
            prefix=f"{layer_name}.layers",
        )
        children.extend(metas)
        return nn.Sequential(*blocks), out_shape, children

    if layer_type == "residual":
        inner_specs = layer_cfg.get("layers")
        if not isinstance(inner_specs, list) or not inner_specs:
            raise ValueError(f"residual 层 {layer_name} 必须包含非空的 layers 列表。")
        blocks, out_shape, metas = build_layers(
            inner_specs,
            in_shape,
            context,
            prefix=f"{layer_name}.layers",
        )
        children.extend(metas)
        inner = nn.Sequential(*blocks)

        projection_mode = str(layer_cfg.get("projection", "auto")).lower()
        projection: nn.Module = nn.Identity()
        if projection_mode == "auto":
            # auto：形状相同用恒等映射；通道/特征数不同时自动选择 1x1 卷积或线性层。
            if in_shape == out_shape:
                projection_mode = "identity"
            else:
                if len(in_shape) == 3 and len(out_shape) == 3:
                    projection_mode = "conv1x1"
                elif len(in_shape) == 1 and len(out_shape) == 1:
                    projection_mode = "linear"
                else:
                    raise ValueError(
                        f"残差层 {layer_name} 无法自动匹配输入 {in_shape} 与输出 {out_shape}；"
                        "请显式指定 projection。"
                    )
        if projection_mode in {"conv1x1", "conv"}:
            if len(in_shape) != 3 or len(out_shape) != 3:
                raise ValueError(f"残差层 {layer_name} 的 conv1x1 projection 只适用于 [C,H,W] 特征。")
            projection_conv = nn.Conv2d(
                int(in_shape[0]),
                int(out_shape[0]),
                kernel_size=1,
                stride=1,
                bias=bool(layer_cfg.get("projection_bias", False)),
            )
            projection_norm = _build_norm(
                layer_cfg.get("projection_norm"),
                out_shape,
                f"{layer_name}.projection",
            )
            projection = nn.Sequential(projection_conv, projection_norm)
        elif projection_mode == "linear":
            if len(in_shape) != 1 or len(out_shape) != 1:
                raise ValueError(f"残差层 {layer_name} 的 linear projection 只适用于 2 维特征。")
            projection = nn.Linear(
                int(in_shape[0]),
                int(out_shape[0]),
                bias=bool(layer_cfg.get("projection_bias", True)),
            )
        elif projection_mode == "identity" and in_shape != out_shape:
            raise ValueError(
                f"残差层 {layer_name} 的 projection=identity，但输入 {in_shape} 与输出 {out_shape} 不同。"
            )
        elif projection_mode == "none" and in_shape != out_shape:
            raise ValueError(
                f"残差层 {layer_name} 的输入 {in_shape} 与输出 {out_shape} 不同，"
                "projection=none 无法相加；请使用 projection=auto 或 conv1x1/linear。"
            )
        elif projection_mode not in {"auto", "identity", "none", "conv1x1", "conv", "linear"}:
            raise ValueError(
                f"残差层 {layer_name} 的 projection='{projection_mode}' 不支持，"
                "可选 auto/identity/conv1x1/linear/none。"
            )

        # 复合层不递归初始化内层；但投影分支应遵循当前层的初始化配置。
        projection_init = layer_cfg.get("projection_weight_init")
        if projection_init is not None:
            _apply_weight_init(projection, projection_init, f"{layer_name}.projection")

        return ResidualCore(inner, projection), out_shape, children

    raise ValueError(f"未知的核心层类型 '{layer_type}'。")

def _build_block_single(
    raw_cfg: Mapping[str, Any],
    in_shape: tuple[int, ...],
    context: BuildContext,
    layer_name: str,
) -> tuple[LayerBlock, tuple[int, ...], dict[str, Any]]:
    """构建一个层块（不处理 repeat）。"""

    cfg = copy.deepcopy(dict(raw_cfg))
    if "type" not in cfg:
        raise ValueError(f"层 {layer_name} 缺少 type 字段。")
    layer_type = _normalize_type(cfg["type"])
    defaults = context.defaults

    # 默认值采用“键不存在才继承”的规则：显式写 null 表示关闭该组件。
    norm_spec = cfg["norm"] if "norm" in cfg else defaults.get("norm")
    activation_spec = cfg["activation"] if "activation" in cfg else defaults.get("activation")
    dropout_spec = cfg["dropout"] if "dropout" in cfg else defaults.get("dropout")
    order = _parse_order(cfg, defaults)
    use_checkpoint = bool(cfg.get("checkpoint", defaults.get("checkpoint", False)))

    # 独立激活层：type: activation, name: relu / gelu ...
    if layer_type == "activation":
        activation_params = {
            key: value
            for key, value in cfg.items()
            if key
            not in {
                "type",
                "repeat",
                "order",
                "trainable",
                "weight_init",
                "checkpoint",
                "dropout",
            }
        }
        if "name" not in activation_params:
            raise ValueError(f"独立激活层 {layer_name} 必须配置 name，例如 name: relu。")
        activation_spec = activation_params
        order = cfg.get("order", ["core", "activation", "dropout"])
        core: nn.Module = nn.Identity()
        norm_spec = None
        out_shape = in_shape

    # 独立 Dropout 层：type: dropout, p: 0.3。
    elif layer_type in {"dropout", "dropout2d", "alpha_dropout"}:
        dropout_spec = {
            "type": layer_type,
            "p": cfg.get("p", cfg.get("probability", 0.5)),
        }
        order = cfg.get("order", ["core"])
        # 独立 Dropout 的 core 就是 Dropout 本体，不能放成 Identity 后再指望
        # order 中的 dropout 部件；order 默认只有 core，否则会被静默跳过。
        core = _build_dropout(dropout_spec, in_shape, layer_name)
        dropout_spec = None
        norm_spec = None
        activation_spec = None
        out_shape = in_shape

    # 独立归一化层：type: batchnorm2d/layernorm/groupnorm...
    elif layer_type in {"batchnorm2d", "batchnorm1d", "instancenorm2d", "layernorm", "groupnorm"}:
        norm_params = {
            key: value
            for key, value in cfg.items()
            if key
            not in {
                "type",
                "repeat",
                "order",
                "trainable",
                "weight_init",
                "checkpoint",
                "activation",
                "dropout",
            }
        }
        norm_params["type"] = layer_type
        core = _build_norm(norm_params, in_shape, layer_name)
        norm_spec = None
        order = cfg.get(
            "order",
            ["core", "activation", "dropout"] if activation_spec is not None else ["core"],
        )
        out_shape = in_shape

    else:
        if isinstance(norm_spec, Mapping):
            norm_type_value = norm_spec.get("type", norm_spec.get("name", ""))
        else:
            norm_type_value = norm_spec
        has_norm = norm_spec is not None and _normalize_type(norm_type_value) not in {
            "",
            "none",
            "identity",
        }
        core, out_shape, children = _build_core(
            layer_type,
            cfg,
            in_shape,
            has_norm,
            context,
            layer_name,
        )
        # children 由外层摘要使用；这里先挂到临时属性，随后返回值带回。
        cfg["_children"] = children

    norm = _build_norm(norm_spec, out_shape, f"{layer_name}.norm")
    activation = _build_activation(activation_spec, f"{layer_name}.activation")
    dropout = _build_dropout(dropout_spec, out_shape, f"{layer_name}.dropout")

    # 非复合层在此统一初始化。复合层的内部层已在递归构建时各自初始化。
    init_spec = cfg.get("weight_init", defaults.get("weight_init"))
    if layer_type not in {"sequence", "residual"}:
        _apply_weight_init(core, init_spec, layer_name)

    block = LayerBlock(
        core=core,
        norm=norm,
        activation=activation,
        dropout=dropout,
        order=order,
        layer_name=layer_name,
        use_checkpoint=use_checkpoint,
    )

    trainable = bool(cfg.get("trainable", True))
    if not trainable:
        for parameter in block.parameters():
            parameter.requires_grad_(False)

    metadata = {
        "name": layer_name,
        "type": layer_type,
        "shape": out_shape,
        "trainable": trainable,
        "checkpoint": use_checkpoint,
        "parameters": sum(parameter.numel() for parameter in block.parameters()),
        "children": cfg.get("_children", []),
    }
    return block, out_shape, metadata


def build_layers(
    layer_specs: list[Mapping[str, Any]],
    in_shape: tuple[int, ...],
    context: BuildContext,
    prefix: str,
) -> tuple[list[nn.Module], tuple[int, ...], list[dict[str, Any]]]:
    """按列表顺序构建层块，并自动传递形状。

    repeat 表示把同一份层配置连续构建多次。对卷积层而言，这只适合
    输出通道固定的情况；需要逐层改变通道数时应写多个独立配置项。
    """

    blocks: list[nn.Module] = []
    metadata: list[dict[str, Any]] = []
    current_shape = in_shape

    for index, raw in enumerate(layer_specs):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{prefix}[{index}] 必须是字典形式的层配置。")
        cfg = copy.deepcopy(dict(raw))
        repeat = int(cfg.pop("repeat", 1))
        if repeat < 1:
            raise ValueError(f"{prefix}[{index}].repeat 必须 >= 1。")
        for copy_index in range(repeat):
            suffix = f"[{index}]" if repeat == 1 else f"[{index}:{copy_index}]"
            layer_name = f"{prefix}{suffix}"
            block, current_shape, meta = _build_block_single(
                cfg,
                current_shape,
                context,
                layer_name,
            )
            blocks.append(block)
            metadata.append(meta)

    return blocks, current_shape, metadata

