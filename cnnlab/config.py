"""YAML 配置的加载、合并、覆盖、校验与快照保存。

设计目标：
1. 所有可调参数都从配置文件读取，Python 代码中不散落魔法数字。
2. configs/default.yaml 给出完整默认值；场景配置只写需要修改的字段。
3. 命令行 --set 可以临时覆盖少量参数，但会再次经过同一套校验。
4. 每次训练把最终配置保存到运行目录，保证实验可追溯。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import PROJECT_ROOT, save_json


# 默认配置固定放在项目 configs 目录下，所有场景配置都继承它。
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigError(ValueError):
    """配置内容不合法时抛出的专用异常，便于命令行给出清晰提示。"""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，返回新字典，不修改输入对象。

    规则：
    - 两边都是字典：继续递归合并。
    - override 中的列表/标量：完整替换 base 中的值。
      这样可以删除默认列表中的元素，而不会被意外拼接。
    - override 中的 None：保留为 None，表示显式关闭该选项。
    """

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_yaml(path: Path) -> dict[str, Any]:
    """读取一个 YAML 文件，并确保顶层是字典。"""

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射（键值对）：{path}")
    return data


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """按照 a.b.c 的形式设置嵌套配置值。

    只允许覆盖 default.yaml 中已经存在的键，从而及时捕获拼写错误。
    例如把 train.epochs 写成 train.epoch 会直接报错，而不是悄悄无效。
    """

    keys = dotted_key.split(".")
    node: Any = config
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"覆盖配置失败：不存在配置项 '{dotted_key}'。")
        node = node[key]
    final_key = keys[-1]
    if not isinstance(node, dict) or final_key not in node:
        raise ConfigError(f"覆盖配置失败：不存在配置项 '{dotted_key}'。")
    node[final_key] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> None:
    """应用命令行传入的 key=value 覆盖项。

    value 使用 YAML 语法解析，因此可以写：
    - train.epochs=5
    - dataset.batch_size=64
    - model.layers[0].activation=null
    注意：复杂列表通常更适合直接修改 YAML 场景配置。
    """

    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"覆盖项 '{item}' 格式错误，应写成 键=值，例如 train.epochs=10。")
        key, raw_value = item.split("=", maxsplit=1)
        key = key.strip()
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"覆盖项 '{item}' 的值无法按 YAML 解析：{exc}") from exc
        _set_dotted(config, key, value)


def _require(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    """读取必需键；缺失时给出包含完整路径的中文错误。"""

    if key not in mapping:
        raise ConfigError(f"配置缺少必需项：{location}.{key}")
    return mapping[key]


def _check_positive_int(value: Any, location: str, allow_zero: bool = False) -> None:
    """校验整数配置项，避免在训练深处才因负数或浮点数报错。"""

    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{location} 必须是大于等于 {minimum} 的整数，当前为 {value!r}。")


def validate_config(config: dict[str, Any]) -> None:
    """对跨模块配置做统一校验。

    这里检查的是通用且高价值的约束；某个具体层/优化器的参数会由相应
    构建函数再次校验。把明显错误提前到训练开始前发现，能节省大量时间。
    """

    for section in (
        "dataset",
        "model",
        "runtime",
        "train",
        "loss",
        "optimizer",
        "scheduler",
        "checkpoint",
        "logging",
        "evaluation",
        "predict",
    ):
        if section not in config or not isinstance(config[section], Mapping):
            raise ConfigError(f"配置缺少顶层配置段：{section}")

    dataset = config["dataset"]
    dataset_name = str(_require(dataset, "name", "dataset")).lower()
    supported_datasets = {
        "mnist",
        "fashion_mnist",
        "kmnist",
        "cifar10",
        "cifar100",
        "image_folder",
        "synthetic",
        "spectra",
    }
    if dataset_name not in supported_datasets:
        raise ConfigError(
            f"dataset.name='{dataset_name}' 不受支持，可选：{', '.join(sorted(supported_datasets))}。"
        )
    _check_positive_int(dataset.get("batch_size"), "dataset.batch_size")
    _check_positive_int(dataset.get("num_workers"), "dataset.num_workers", allow_zero=True)
    _check_positive_int(dataset.get("in_channels"), "dataset.in_channels")
    image_size = dataset.get("image_size")
    if isinstance(image_size, int) and not isinstance(image_size, bool):
        if image_size <= 0:
            raise ConfigError("dataset.image_size 必须大于 0。")
    elif isinstance(image_size, (list, tuple)) and len(image_size) in {1, 2}:
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in image_size):
            raise ConfigError("dataset.image_size 列表中的每一项都必须是正整数。")
    else:
        raise ConfigError("dataset.image_size 必须是正整数或长度 1/2 的正整数列表。")
    configured_classes = dataset.get("num_classes")
    if configured_classes not in {None, "auto"}:
        if not isinstance(configured_classes, int) or isinstance(configured_classes, bool):
            raise ConfigError("dataset.num_classes 必须是整数、auto 或 null。")
        if configured_classes < 1:
            raise ConfigError("dataset.num_classes 必须至少为 1。")
    if dataset_name not in {"image_folder", "synthetic", "spectra"} and configured_classes in {
        None,
        "auto",
    }:
        raise ConfigError(f"{dataset_name} 必须配置 dataset.num_classes。")

    model = config["model"]
    layers = model.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ConfigError("model.layers 必须是非空列表，至少定义一个网络层。")
    head = model.get("head", [])
    if not isinstance(head, list):
        raise ConfigError("model.head 必须是列表，可以为空列表 []。")
    input_shape = model.get("input_shape")
    if input_shape is not None:
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ConfigError("model.input_shape 必须是 [通道数, 高, 宽] 形式的列表或 null。")
        if any(int(size) <= 0 for size in input_shape):
            raise ConfigError("model.input_shape 中每个维度都必须大于 0。")

    runtime = config["runtime"]
    if str(runtime.get("device", "auto")) not in {"auto", "cpu", "cuda", "mps"} and not str(
        runtime.get("device", "")
    ).startswith("cuda:"):
        raise ConfigError("runtime.device 可选 auto/cpu/cuda/cuda:N/mps。")
    _check_positive_int(runtime.get("seed"), "runtime.seed", allow_zero=True)

    train = config["train"]
    _check_positive_int(train.get("epochs"), "train.epochs", allow_zero=True)
    _check_positive_int(train.get("grad_accumulation_steps"), "train.grad_accumulation_steps")
    if int(train.get("print_freq", 1)) < 1:
        raise ConfigError("train.print_freq 必须大于等于 1。")
    monitor = str(train.get("monitor", "val_loss"))
    if monitor not in {
        "val_loss",
        "val_accuracy",
        "val_f1_macro",
        "train_loss",
        "train_accuracy",
    }:
        raise ConfigError(
            "train.monitor 可选 val_loss/val_accuracy/val_f1_macro/train_loss/train_accuracy。"
        )

    loss = config["loss"]
    supported_losses = {
        "cross_entropy",
        "bce_with_logits",
        "nll",
    }
    if str(loss.get("name")).lower() not in supported_losses:
        raise ConfigError(f"loss.name 可选：{', '.join(sorted(supported_losses))}。")

    optimizer = config["optimizer"]
    supported_optimizers = {"sgd", "adam", "adamw", "rmsprop", "adagrad", "adadelta"}
    if str(optimizer.get("name")).lower() not in supported_optimizers:
        raise ConfigError(f"optimizer.name 可选：{', '.join(sorted(supported_optimizers))}。")
    if float(optimizer.get("lr", 0.0)) <= 0:
        raise ConfigError("optimizer.lr 必须大于 0。")

    scheduler = config["scheduler"]
    supported_schedulers = {
        "none",
        "step",
        "multi_step",
        "cosine",
        "cosine_restarts",
        "plateau",
        "linear",
        "exponential",
        "one_cycle",
    }
    if str(scheduler.get("name")).lower() not in supported_schedulers:
        raise ConfigError(f"scheduler.name 可选：{', '.join(sorted(supported_schedulers))}。")
    if str(scheduler.get("name")).lower() == "plateau" and str(scheduler.get("step_on")) == "batch":
        raise ConfigError("ReduceLROnPlateau 只支持按 epoch 更新，请把 scheduler.step_on 设为 epoch。")
    if str(scheduler.get("name")).lower() == "one_cycle" and str(scheduler.get("step_on")) != "batch":
        raise ConfigError(
            "OneCycleLR 需要按 batch 更新，请把 scheduler.step_on 设为 batch。"
        )

    if dataset_name != "image_folder" and configured_classes not in {None, "auto"}:
        configured_classes = int(configured_classes)
        if model.get("input_shape") is not None:
            in_channels = int(model["input_shape"][0])
            if dataset_name == "synthetic":
                synthetic_channels = (dataset.get("synthetic") or {}).get(
                    "channels",
                    dataset.get("in_channels", in_channels),
                )
                dataset_channels = int(synthetic_channels)
            else:
                dataset_channels = int(dataset.get("in_channels", in_channels))
            if in_channels != dataset_channels:
                raise ConfigError(
                    "model.input_shape 的通道数与 dataset.in_channels 不一致："
                    f"{in_channels} != {dataset_channels}。"
                )
        if configured_classes < 1:
            raise ConfigError("dataset.num_classes 必须至少为 1。")


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """加载场景配置并继承默认配置。

    如果 path 就是 default.yaml，则直接使用；否则执行“默认配置 <- 场景配置
    <- 命令行覆盖”的三级合并。返回的是普通 dict，后续模块可自由读取。
    """

    config_path = Path(path)
    if not config_path.is_absolute():
        # 允许在项目根目录下直接写 configs/xxx.yaml。
        candidate = PROJECT_ROOT / config_path
        config_path = candidate if candidate.exists() else config_path.resolve()

    scenario = read_yaml(config_path)
    if config_path.resolve() == DEFAULT_CONFIG_PATH.resolve():
        config = copy.deepcopy(scenario)
    else:
        config = deep_merge(read_yaml(DEFAULT_CONFIG_PATH), scenario)

    apply_overrides(config, overrides)
    if validate:
        validate_config(config)

    # 派生信息只在内存中使用；保存配置时去掉下划线开头的内部键。
    config["_meta"] = {
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config_path.resolve()),
    }
    return config


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """去掉以下划线开头的内部字段，得到适合写入 YAML 的纯配置。"""

    return {key: copy.deepcopy(value) for key, value in config.items() if not str(key).startswith("_")}


def save_config_snapshot(config: Mapping[str, Any], run_dir: Path) -> Path:
    """把最终生效配置保存到运行目录。

    YAML 快照便于人工阅读；JSON 快照便于其他程序精确读取。
    两者都保存，避免以后只看到场景配置而不知道默认值是什么。
    """

    payload = public_config(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = run_dir / "resolved_config.yaml"
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)
    save_json(payload, run_dir / "resolved_config.json")
    return yaml_path


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套配置展开成 a.b.c 的键值形式，用于日志摘要。"""

    result: dict[str, Any] = {}
    for key, value in config.items():
        if str(key).startswith("_"):
            continue
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_config(value, full_key))
        else:
            result[full_key] = value
    return result

