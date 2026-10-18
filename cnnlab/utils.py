"""项目通用工具函数。

这里集中放置随机种子、设备选择、日志、JSON 读写、参数量统计等与具体
模型无关的工具，避免训练脚本、评估脚本各写一份而产生行为差异。
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


# 项目根目录：cnnlab/utils.py 的上两级，即 cnn_framework/。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path_value: str | os.PathLike[str] | None, base: Path | None = None) -> Path | None:
    """把配置中的路径解析为绝对路径。

    规则：
    1. 空值保持为空，便于表示“不启用某功能”。
    2. 绝对路径原样返回。
    3. 相对路径默认相对于项目根目录，而不是当前命令行所在目录。
       这样从任意工作目录启动脚本都能得到一致结果。
    """

    if path_value is None or str(path_value).strip() == "":
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return ((base or PROJECT_ROOT) / path).resolve()


def set_seed(seed: int, deterministic: bool = False, benchmark: bool = False) -> None:
    """固定 Python、NumPy、PyTorch 与 CUDA 的随机种子。

    参数说明：
    - seed：随机种子。相同种子、相同软硬件与相同数据顺序下，结果应接近。
    - deterministic：是否请求 PyTorch 的确定性算法。部分 CUDA 算子没有确定性实现，
      因此使用 warn_only=True，遇到限制时给出警告而不是直接中断训练。
    - benchmark：非确定性模式下是否让 cuDNN 自动搜索最快卷积算法。
      输入尺寸固定时通常能提速，但每次运行的具体算法可能不同。
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cudnn.benchmark=True 会在运行时选择算法，和确定性模式冲突。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # 兼容不支持 warn_only 参数的旧版 PyTorch。
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = bool(benchmark)


def resolve_device(requested: str = "auto", logger: logging.Logger | None = None) -> torch.device:
    """根据配置选择计算设备，并在回退时明确记录原因。"""

    value = str(requested or "auto").strip().lower()
    if value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif value == "cuda":
        device = torch.device("cuda")
    elif value.startswith("cuda:"):
        index = int(value.split(":", maxsplit=1)[1])
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"配置要求使用 {value}，但当前只检测到 {torch.cuda.device_count()} 块 CUDA 显卡。"
            )
        device = torch.device(value)
    elif value == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise ValueError("配置要求使用 MPS，但当前 PyTorch 环境不支持 MPS。")
        device = torch.device("mps")
    elif value == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"未知设备 '{requested}'，可选 auto/cpu/cuda/cuda:N/mps。")

    if device.type == "cuda" and not torch.cuda.is_available():
        if logger is not None:
            logger.warning("配置要求使用 CUDA，但当前 CUDA 不可用，已回退到 CPU。")
        device = torch.device("cpu")

    if logger is not None:
        logger.info("使用计算设备：%s", device)
        if device.type == "cuda":
            logger.info(
                "GPU：%s，显存 %.2f GB",
                torch.cuda.get_device_name(device),
                torch.cuda.get_device_properties(device).total_memory / 1024**3,
            )
    return device


def setup_logger(
    run_dir: Path,
    log_file: str = "train.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    logger_name: str = "cnnlab",
) -> logging.Logger:
    """创建同时输出到控制台和 UTF-8 日志文件的 Logger。

    每次调用先移除旧 handler，避免多次运行同一进程时日志重复。
    文件使用 UTF-8 编码，确保中文日志在 Windows 记事本/VSCode 中正常显示。
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, str(console_level).upper(), logging.INFO))
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(run_dir / log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(getattr(logging, str(file_level).upper(), logging.DEBUG))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _json_default(value: Any) -> Any:
    """把常见科学计算对象转换为可写入 JSON 的 Python 基础类型。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """以 UTF-8 保存 JSON，并自动创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent, default=_json_default)


def load_json(path: Path, default: Any = None) -> Any:
    """读取 JSON；文件不存在时返回 default。"""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数量。

    trainable_only=True 时只统计 requires_grad=True 的参数，
    适合观察冻结骨干网络后真正需要训练的参数量。
    """

    parameters: Iterable[torch.nn.Parameter] = model.parameters()
    return sum(
        parameter.numel()
        for parameter in parameters
        if (parameter.requires_grad or not trainable_only)
    )


def format_number(number: int | float) -> str:
    """把较大的参数量转换成易读的 K/M/B 形式。"""

    value = float(number)
    for unit in ("", "K", "M", "B"):
        if abs(value) < 1000.0 or unit == "B":
            return f"{value:.2f}{unit}" if unit else f"{int(value)}"
        value /= 1000.0
    return str(number)


def format_seconds(seconds: float) -> str:
    """把秒数格式化为方便日志阅读的 时:分:秒。"""

    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def sanitize_name(value: str, fallback: str = "run") -> str:
    """清理运行名中的文件系统非法字符，避免创建目录失败。"""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" ._")
    return cleaned or fallback


def create_run_dir(
    base_dir: Path,
    prefix: str,
    run_name: str | None = None,
    unique: bool = True,
) -> Path:
    """创建唯一训练输出目录，例如 outputs/runs/cifar10_20261018_201500_resnet。

    即使同一秒启动两次，也会通过 _01、_02 后缀保证目录不冲突，
    从而不会覆盖之前的日志和权重。
    """

    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_prefix = sanitize_name(prefix, "experiment")
    safe_name = sanitize_name(run_name, "") if run_name else ""
    stem = f"{safe_prefix}_{timestamp}" + (f"_{safe_name}" if safe_name else "")

    candidate = base_dir / stem
    if unique:
        index = 1
        while candidate.exists():
            candidate = base_dir / f"{stem}_{index:02d}"
            index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def seed_worker(worker_id: int) -> None:
    """为 DataLoader 子进程设置独立但可复现的随机种子。

    worker_id 参数由 PyTorch 自动传入；即使函数体暂时没用到它，
    也必须保留这个签名。NumPy 种子取自 PyTorch 为当前 worker 生成的种子，
    确保数据增强操作同样可以复现。
    """

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_matplotlib_cache(directory: Path) -> None:
    """把 Matplotlib 字体缓存放到运行目录，避免系统用户目录不可写。

    Matplotlib 默认会把缓存写到用户目录；在受限环境或共享机器上可能
    出现 Permission denied 警告。必须在导入 matplotlib.pyplot 之前设置。
    """

    directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(directory)


def describe_environment() -> dict[str, Any]:
    """收集运行环境信息，写入实验目录，便于以后复现问题。"""

    info: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = None
    return info

