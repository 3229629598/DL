"""光谱数据预处理、热图生成与光谱 CNN 数据集。

处理流程严格对应需求图：

    气体原始光谱 - 暗电流 -> z-score 归一化 \
                                                -> 相减 -> 平滑 -> 材料×波长矩阵
    空白原始光谱 - 暗电流 -> z-score 归一化 /

原始数据中每个材料的 txt 包含 50 条扫描；部分 xlsx 只有 45~49 列，
因此本模块以 txt 为光谱数值主来源，xlsx 只用于读取 dark_avg。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import PROJECT_ROOT, resolve_path, save_json, sanitize_name


@dataclass
class SpectralCache:
    """缓存中保存的数据与元信息。"""

    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    replicates: np.ndarray
    wavelengths: np.ndarray
    materials: list[str]
    class_names: list[str]
    scan_count: int
    metadata: dict[str, Any]
    cache_path: Path


def _decode_text(path: Path) -> str:
    """光谱仪 txt 常见编码为 GBK；依次尝试常见编码。"""

    raw = path.read_bytes()
    for encoding in ("gbk", "utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("gbk", errors="replace")


def read_scan_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取一个光谱仪 txt，返回 wavelengths[n_wavelength] 和 scans[50,n_wavelength]。

    文件结构：
    - 一行波长；
    - 后续每行以“时间戳 + 仪器编号”开头，再跟完整光谱强度。
    """

    text = _decode_text(path)
    lines = text.splitlines()
    try:
        begin = next(
            index
            for index, line in enumerate(lines)
            if "Begin Spectral Data" in line
        )
    except StopIteration as exc:
        raise ValueError(f"{path} 中找不到 'Begin Spectral Data'。") from exc

    if begin + 1 >= len(lines):
        raise ValueError(f"{path} 没有波长行。")
    wavelengths = np.array(
        [float(token) for token in lines[begin + 1].split() if _is_float(token)],
        dtype=np.float64,
    )
    if wavelengths.size < 10:
        raise ValueError(f"{path} 的波长数量异常：{wavelengths.size}。")

    scans: list[list[float]] = []
    for line in lines[begin + 2 :]:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        # 前两列是时间戳和仪器编号，不属于光谱。
        values: list[float] = []
        for token in tokens[2:]:
            if not _is_float(token):
                continue
            values.append(float(token))
            if len(values) == wavelengths.size:
                break
        if len(values) == wavelengths.size:
            scans.append(values)

    if not scans:
        raise ValueError(f"{path} 没有解析到光谱扫描。")
    return wavelengths, np.asarray(scans, dtype=np.float32)


def _is_float(value: str) -> bool:
    """判断字符串能否转换成 float，避免时间戳字段进入光谱数组。"""

    try:
        float(value)
        return True
    except ValueError:
        return False


def read_dark_avg(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 dark_avg.xlsx 的波长和暗电流两列。"""

    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    wavelengths: list[float] = []
    values: list[float] = []
    for row in sheet.iter_rows(values_only=True):
        if row[0] is None or row[1] is None:
            continue
        wavelengths.append(float(row[0]))
        values.append(float(row[1]))
    workbook.close()
    return np.asarray(wavelengths, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _material_name(stem: str, aliases: Mapping[str, str]) -> str:
    """把文件名主干规范成材料名，并应用配置中的别名。"""

    return str(aliases.get(stem, stem))


def _material_files(folder: Path, aliases: Mapping[str, str]) -> dict[str, Path]:
    """扫描一个气味目录下的 txt，建立材料名到 txt 的映射。"""

    result: dict[str, Path] = {}
    for path in sorted(folder.glob("*.txt")):
        # 文件名形如 808-309_QEP058691_23-19-49-755.txt，
        # 材料名只取 _QEP 之前的稳定前缀。
        material_stem = path.stem.split("_QEP", maxsplit=1)[0]
        name = _material_name(material_stem, aliases)
        if name in result:
            raise ValueError(
                f"目录 {folder} 中的材料名重复：{name}；"
                f"文件为 {result[name].name} 和 {path.name}。"
            )
        result[name] = path
    return result


def _select_wavelengths(
    dark_wavelengths: np.ndarray,
    dark_values: np.ndarray,
    wavelength_min: float | None,
    wavelength_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """按配置裁切 dark_avg 的波长区间，作为所有样本的统一波长轴。"""

    mask = np.ones_like(dark_wavelengths, dtype=bool)
    if wavelength_min is not None:
        mask &= dark_wavelengths >= float(wavelength_min)
    if wavelength_max is not None:
        mask &= dark_wavelengths <= float(wavelength_max)
    target = dark_wavelengths[mask]
    values = dark_values[mask]
    if target.size < 10:
        raise ValueError(
            f"裁剪后的波长点数过少：{target.size}。请检查 wavelength_min/max。"
        )
    return target, values


def _resample_scan(
    wavelengths: np.ndarray,
    scans: np.ndarray,
    target_wavelengths: np.ndarray,
) -> np.ndarray:
    """把 txt 的 1044 个波长点插值到 dark_avg 的 592 个波长点。"""

    if target_wavelengths.min() < wavelengths.min() or target_wavelengths.max() > wavelengths.max():
        raise ValueError(
            "目标波长范围超出 txt 波长范围："
            f"txt=[{wavelengths.min():.3f},{wavelengths.max():.3f}]，"
            f"target=[{target_wavelengths.min():.3f},{target_wavelengths.max():.3f}]。"
        )
    result = np.empty((scans.shape[0], target_wavelengths.size), dtype=np.float32)
    for index, scan in enumerate(scans):
        result[index] = np.interp(target_wavelengths, wavelengths, scan).astype(np.float32)
    return result


def _zscore(scan: np.ndarray, axis: int) -> np.ndarray:
    """按指定轴做 z-score；标准差为 0 时使用 eps 防止除零。"""

    mean = scan.mean(axis=axis, keepdims=True)
    std = scan.std(axis=axis, keepdims=True)
    return (scan - mean) / np.maximum(std, 1.0e-8)


def _moving_average(scan: np.ndarray, window: int) -> np.ndarray:
    """沿波长轴做边缘填充的移动平均。"""

    if window <= 1:
        return scan
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(scan, ((0, 0), (pad, pad)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    result = np.empty_like(scan)
    for index, row in enumerate(padded):
        result[index] = np.convolve(row, kernel, mode="valid").astype(np.float32)
    return result


def _process_spectrum_pair(
    gas_path: Path,
    blank_path: Path,
    dark_values: np.ndarray,
    target_wavelengths: np.ndarray,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """执行“减暗电流 -> z-score -> 减空白 -> 平滑”的完整流程。

    返回 [scan_count, wavelength_count]，每一行是一条材料差谱。
    """

    gas_w, gas_scans = read_scan_txt(gas_path)
    blank_w, blank_scans = read_scan_txt(blank_path)
    expected_scans = int(cfg.get("scan_count", 50))
    if gas_scans.shape[0] < expected_scans:
        raise ValueError(f"{gas_path} 只有 {gas_scans.shape[0]} 条扫描，少于 {expected_scans}。")
    if blank_scans.shape[0] < expected_scans:
        raise ValueError(f"{blank_path} 只有 {blank_scans.shape[0]} 条扫描，少于 {expected_scans}。")

    gas = _resample_scan(gas_w, gas_scans[:expected_scans], target_wavelengths)
    blank = _resample_scan(blank_w, blank_scans[:expected_scans], target_wavelengths)
    gas = gas - dark_values[None, :].astype(np.float32)
    blank = blank - dark_values[None, :].astype(np.float32)

    normalize_axis = str(cfg.get("normalize_axis", "wavelength")).lower()
    if normalize_axis == "wavelength":
        gas = _zscore(gas, axis=1)
        blank = _zscore(blank, axis=1)
    elif normalize_axis == "scan":
        gas = _zscore(gas, axis=0)
        blank = _zscore(blank, axis=0)
    elif normalize_axis != "none":
        raise ValueError("spectra.normalize_axis 可选 wavelength/scan/none。")

    blank_mode = str(cfg.get("blank_mode", "mean")).lower()
    if blank_mode == "mean":
        reference = blank.mean(axis=0, keepdims=True)
        difference = gas - reference
    elif blank_mode == "paired":
        difference = gas - blank
    else:
        raise ValueError("spectra.blank_mode 可选 mean/paired。")

    window = int(cfg.get("smooth_window", 9))
    difference = _moving_average(difference, window)
    clip_value = cfg.get("clip")
    if clip_value is not None:
        difference = np.clip(difference, -float(clip_value), float(clip_value))
    if not np.isfinite(difference).all():
        raise ValueError(f"{gas_path} 处理结果包含 NaN/Inf。")
    return difference.astype(np.float32)


def _raw_signature(raw_dir: Path, pattern: str = "*.txt") -> list[list[Any]]:
    """生成原始文件的轻量签名，用于判断缓存是否过期。"""

    signature: list[list[Any]] = []
    for path in sorted(raw_dir.rglob(pattern)):
        stat = path.stat()
        signature.append(
            [
                str(path.relative_to(raw_dir)),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            ]
        )
    return signature


def _config_fingerprint(
    raw_dir: Path,
    dark_path: Path,
    cfg: Mapping[str, Any],
) -> str:
    """对处理参数和原始文件时间戳做哈希，配置或数据变了就自动重建。"""

    payload = {
        "raw_dir": str(raw_dir),
        "dark_path": str(dark_path),
        "cfg": cfg,
        "raw_signature": _raw_signature(raw_dir),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _discover_materials(
    raw_dir: Path,
    aliases: Mapping[str, str],
    strategy: str,
    material_order: Sequence[str] | None,
) -> tuple[list[str], dict[str, dict[str, Path]], dict[str, Path], list[str]]:
    """发现气味、材料和 none 文件，并按策略决定统一材料集合。"""

    odor_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir() and path.name != "none")
    none_dir = raw_dir / "none"
    if not odor_dirs:
        raise FileNotFoundError(f"{raw_dir} 下没有气味目录。")
    if not none_dir.exists():
        raise FileNotFoundError(f"找不到空白光谱目录：{none_dir}")

    odor_files: dict[str, dict[str, Path]] = {}
    all_sets: list[set[str]] = []
    for odor_dir in odor_dirs:
        files = _material_files(odor_dir, aliases)
        if not files:
            raise FileNotFoundError(f"气味目录中没有 txt：{odor_dir}")
        odor_files[odor_dir.name] = files
        all_sets.append(set(files))

    none_files = _material_files(none_dir, aliases)
    if not none_files:
        raise FileNotFoundError(f"none 目录中没有 txt：{none_dir}")
    all_sets.append(set(none_files))

    if strategy == "intersection":
        materials = set.intersection(*all_sets)
    elif strategy == "union":
        materials = set.union(*all_sets)
    else:
        raise ValueError("spectra.material_strategy 可选 intersection/union。")
    if not materials:
        raise ValueError("材料交集为空，请检查文件命名或 material_aliases。")

    if material_order:
        ordered = [str(name) for name in material_order if str(name) in materials]
    else:
        ordered = sorted(materials)
    # 配置中的 material_order 可能只列出了部分材料；剩余材料按名称补在后面。
    ordered.extend(sorted(materials - set(ordered)))
    return ordered, odor_files, none_files, [path.name for path in odor_dirs]


def _cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "spectral_dataset.npz", cache_dir / "spectral_meta.json"


def _load_cache(cache_path: Path, meta_path: Path) -> SpectralCache:
    """从 npz 和 JSON 恢复缓存。"""

    data = np.load(cache_path, allow_pickle=False)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return SpectralCache(
        x=data["x"],
        y=data["y"],
        groups=data["groups"].astype(str),
        replicates=data["replicates"],
        wavelengths=data["wavelengths"],
        materials=[str(value) for value in data["materials"].tolist()],
        class_names=[str(value) for value in data["class_names"].tolist()],
        scan_count=int(metadata["scan_count"]),
        metadata=metadata,
        cache_path=cache_path,
    )


def _save_cache(cache: SpectralCache, cache_path: Path, meta_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=cache.x,
        y=cache.y,
        groups=cache.groups.astype("U"),
        replicates=cache.replicates,
        wavelengths=cache.wavelengths,
        materials=np.asarray(cache.materials, dtype="U"),
        class_names=np.asarray(cache.class_names, dtype="U"),
    )
    meta_path.write_text(
        json.dumps(cache.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _setup_chinese_font() -> None:
    """让热图标题、材料名和坐标轴能够显示中文。"""

    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def _heatmap_limits(mean_diff: np.ndarray, cfg: Mapping[str, Any]) -> tuple[float, float]:
    """计算热图色标范围；auto 使用对称的稳健分位数。"""

    vmin = cfg.get("vmin", -4.0)
    vmax = cfg.get("vmax", 4.0)
    if str(vmin).lower() == "auto" or str(vmax).lower() == "auto":
        finite = np.abs(mean_diff[np.isfinite(mean_diff)])
        limit = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        limit = max(limit, 1.0)
        return -limit, limit
    return float(vmin), float(vmax)


def save_heatmaps(
    cache: SpectralCache,
    cfg: Mapping[str, Any],
    logger: Any = None,
    force: bool = False,
) -> dict[str, str]:
    """为每个气味生成“材料 × 波长”的平均差谱热图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _setup_chinese_font()
    heat_cfg = cfg.get("heatmaps") or {}
    if not bool(heat_cfg.get("enabled", True)):
        return {}
    output_dir = resolve_path(heat_cfg.get("output_dir", "outputs/spectra/heatmaps"), PROJECT_ROOT)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    x = cache.x.reshape(len(cache.class_names), cache.scan_count, len(cache.materials), -1)
    # x 形状 [气味, 扫描, 材料, 波长]；平均扫描后得到 [气味, 材料, 波长]。
    mean_diff = x.mean(axis=1)
    materials = cache.materials
    wavelengths = cache.wavelengths
    vmin, vmax = _heatmap_limits(mean_diff, heat_cfg)
    cmap = str(heat_cfg.get("cmap", "RdBu_r"))
    figsize = tuple(float(value) for value in heat_cfg.get("figsize", [18, 12]))
    dpi = int(heat_cfg.get("dpi", 200))
    overwrite = bool(heat_cfg.get("overwrite", False))
    index: dict[str, str] = {}

    for odor_index, odor in enumerate(cache.class_names):
        output_path = output_dir / f"{sanitize_name(odor)}.png"
        index[odor] = str(output_path)
        if output_path.exists() and not overwrite and not force:
            continue
        figure, axis = plt.subplots(figsize=figsize)
        image = axis.imshow(
            mean_diff[odor_index],
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            extent=[float(wavelengths[0]), float(wavelengths[-1]), len(materials) - 0.5, -0.5],
        )
        axis.set_yticks(range(len(materials)))
        axis.set_yticklabels(materials, fontsize=8)
        tick_values = heat_cfg.get("x_ticks")
        if tick_values:
            tick_values = [float(value) for value in tick_values]
        else:
            middle_target = 618.185
            middle_index = int(np.abs(wavelengths - middle_target).argmin())
            tick_values = [float(wavelengths[0]), float(wavelengths[middle_index]), float(wavelengths[-1])]
        axis.set_xticks(tick_values)
        axis.set_xticklabels([f"{value:.3f}" for value in tick_values])
        axis.set_xlabel("波长（nm）", fontsize=12)
        axis.set_ylabel("材料", fontsize=12)
        axis.set_title(odor, fontsize=18, pad=12)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("相对光强", fontsize=12)
        figure.tight_layout()
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        if logger is not None and (odor_index + 1) % 10 == 0:
            logger.info("热图进度：%d/%d", odor_index + 1, len(cache.class_names))

    save_json(index, output_dir / "heatmap_index.json")
    np.savez_compressed(
        output_dir / "mean_diff_spectra.npz",
        mean_diff=mean_diff,
        wavelengths=wavelengths,
        materials=np.asarray(materials, dtype="U"),
        class_names=np.asarray(cache.class_names, dtype="U"),
    )
    if bool(heat_cfg.get("combined", True)):
        _save_combined_heatmap(mean_diff, cache.class_names, wavelengths, materials, vmin, vmax, cmap, output_dir)
    return index


def _save_combined_heatmap(
    mean_diff: np.ndarray,
    class_names: Sequence[str],
    wavelengths: np.ndarray,
    materials: Sequence[str],
    vmin: float,
    vmax: float,
    cmap: str,
    output_dir: Path,
) -> None:
    """生成一张包含所有气味缩略热图的索引图。"""

    import matplotlib.pyplot as plt

    columns = 4
    rows = math.ceil(len(class_names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.2 * rows), squeeze=False)
    for index in range(rows * columns):
        axis = axes[index // columns][index % columns]
        if index >= len(class_names):
            axis.axis("off")
            continue
        axis.imshow(
            mean_diff[index],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(class_names[index], fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("所有气味：材料 × 波长差谱热图", fontsize=18)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(output_dir / "all_odors_heatmaps.png", dpi=120, bbox_inches="tight")
    plt.close(figure)


def prepare_spectral_cache(
    config: Mapping[str, Any],
    logger: Any = None,
    force_rebuild: bool = False,
) -> SpectralCache:
    """构建或读取光谱数据集缓存，并确保热图存在。"""

    dataset_cfg = config["dataset"]
    cfg = dict(dataset_cfg.get("spectra") or {})
    raw_dir = resolve_path(cfg.get("raw_dir", "data/[original data]"), PROJECT_ROOT)
    dark_path = resolve_path(cfg.get("dark_avg", "data/[original data]/dark_avg.xlsx"), PROJECT_ROOT)
    cache_dir = resolve_path(cfg.get("cache_dir", "data/spectral_processed"), PROJECT_ROOT)
    if raw_dir is None or dark_path is None or cache_dir is None:
        raise ValueError("spectra.raw_dir/dark_avg/cache_dir 不能为空。")
    if not raw_dir.exists():
        raise FileNotFoundError(f"光谱原始目录不存在：{raw_dir}")
    if not dark_path.exists():
        raise FileNotFoundError(f"暗电流文件不存在：{dark_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 将 Matplotlib 字体缓存放到项目数据目录，避免用户目录无写权限时报警。
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib_cache"))
    cache_path, meta_path = _cache_paths(cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    aliases = {str(key): str(value) for key, value in (cfg.get("material_aliases") or {}).items()}
    fingerprint = _config_fingerprint(raw_dir, dark_path, cfg)
    if cache_path.exists() and meta_path.exists() and not force_rebuild:
        try:
            old_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if old_metadata.get("fingerprint") == fingerprint:
                cache = _load_cache(cache_path, meta_path)
                if logger is not None:
                    logger.info("读取光谱缓存：%s（%d 个样本）", cache_path, cache.x.shape[0])
                save_heatmaps(cache, cfg, logger)
                return cache
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            if logger is not None:
                logger.warning("光谱缓存读取失败，将重新构建：%s", exc)

    dark_wavelengths, dark_values = read_dark_avg(dark_path)
    target_wavelengths, target_dark = _select_wavelengths(
        dark_wavelengths,
        dark_values,
        cfg.get("wavelength_min"),
        cfg.get("wavelength_max"),
    )
    materials, odor_files, none_files, odor_order = _discover_materials(
        raw_dir,
        aliases,
        str(cfg.get("material_strategy", "intersection")),
        cfg.get("material_order"),
    )
    class_names = sorted(odor_order)
    scan_count = int(cfg.get("scan_count", 50))
    if scan_count < 1:
        raise ValueError("spectra.scan_count 必须大于 0。")
    if logger is not None:
        logger.info(
            "光谱数据：%d 个气味，%d 种材料，%d 个波长点，每个气味 %d 条扫描。",
            len(class_names),
            len(materials),
            target_wavelengths.size,
            scan_count,
        )

    x = np.zeros(
        (len(class_names), scan_count, len(materials), target_wavelengths.size),
        dtype=np.float32,
    )
    warnings: list[str] = []
    for odor_index, odor in enumerate(class_names):
        files = odor_files[odor]
        missing = [material for material in materials if material not in files]
        if missing:
            warnings.append(f"{odor} 缺少材料：{missing}；已按零光谱填充。")
        for material_index, material in enumerate(materials):
            if material not in files:
                continue
            gas_path = files[material]
            if material not in none_files:
                raise FileNotFoundError(f"none 目录缺少材料 {material} 的 txt。")
            x[odor_index, :, material_index, :] = _process_spectrum_pair(
                gas_path,
                none_files[material],
                target_dark,
                target_wavelengths,
                cfg,
            )
        if logger is not None and (odor_index + 1) % 5 == 0:
            logger.info("光谱处理进度：%d/%d", odor_index + 1, len(class_names))

    x_flat = x.reshape(len(class_names) * scan_count, 1, len(materials), target_wavelengths.size)
    y = np.repeat(np.arange(len(class_names), dtype=np.int64), scan_count)
    groups = np.repeat(np.asarray(class_names, dtype="U"), scan_count)
    replicates = np.tile(np.arange(scan_count, dtype=np.int64), len(class_names))
    metadata = {
        "fingerprint": fingerprint,
        "raw_dir": str(raw_dir),
        "dark_avg": str(dark_path),
        "scan_count": scan_count,
        "n_samples": int(x_flat.shape[0]),
        "n_odors": len(class_names),
        "n_materials": len(materials),
        "n_wavelengths": int(target_wavelengths.size),
        "class_names": class_names,
        "materials": materials,
        "wavelengths": target_wavelengths.tolist(),
        "warnings": warnings,
        "processing": {
            "source": "txt",
            "normalize": "z-score",
            "normalize_axis": cfg.get("normalize_axis", "wavelength"),
            "blank_mode": cfg.get("blank_mode", "mean"),
            "smooth_window": int(cfg.get("smooth_window", 9)),
            "wavelength_min": cfg.get("wavelength_min"),
            "wavelength_max": cfg.get("wavelength_max"),
        },
    }
    cache = SpectralCache(
        x=x_flat,
        y=y,
        groups=groups,
        replicates=replicates,
        wavelengths=target_wavelengths,
        materials=materials,
        class_names=class_names,
        scan_count=scan_count,
        metadata=metadata,
        cache_path=cache_path,
    )
    _save_cache(cache, cache_path, meta_path)
    if logger is not None:
        logger.info("光谱数据集缓存已保存：%s", cache_path)
    # 只要缓存因参数变化而重建，就必须重画热图；否则旧材料顺序/旧色标
    # 会和新数据混在一起。
    save_heatmaps(cache, cfg, logger, force=True)
    return cache


class SpectralArrayDataset(Dataset):
    """在内存中的光谱张量上做在线增强。

    每个样本形状为 [1, 材料数, 波长数]；1 是 CNN 的输入通道。
    """

    def __init__(
        self,
        x: np.ndarray,
        targets: Sequence[int],
        indices: Sequence[int],
        augment: Mapping[str, Any] | None = None,
        training: bool = False,
    ) -> None:
        self.x = x
        self.indices = [int(index) for index in indices]
        self.targets = [int(targets[index]) for index in self.indices]
        self.augment_cfg = dict(augment or {})
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample = np.array(self.x[self.indices[index]], dtype=np.float32, copy=True)
        if self.training:
            shift = int(self.augment_cfg.get("random_shift", 0))
            if shift > 0:
                offset = int(np.random.randint(-shift, shift + 1))
                sample = np.roll(sample, offset, axis=-1)
            scale_range = self.augment_cfg.get("scale") or [1.0, 1.0]
            if len(scale_range) == 2 and float(scale_range[0]) != float(scale_range[1]):
                sample *= float(np.random.uniform(float(scale_range[0]), float(scale_range[1])))
            noise_std = float(self.augment_cfg.get("noise_std", 0.0))
            if noise_std > 0:
                sample += np.random.normal(0.0, noise_std, size=sample.shape).astype(np.float32)
            mask_p = float(self.augment_cfg.get("wavelength_mask_p", 0.0))
            if mask_p > 0 and np.random.rand() < mask_p:
                width = max(1, int(self.augment_cfg.get("wavelength_mask_width", 20)))
                start = int(np.random.randint(0, max(1, sample.shape[-1] - width)))
                sample[..., start : start + width] = 0.0
            material_p = float(self.augment_cfg.get("material_dropout_p", 0.0))
            if material_p > 0:
                rows = np.random.rand(sample.shape[1]) < material_p
                sample[:, rows, :] = 0.0
        return torch.from_numpy(sample), self.targets[index]


def _split_indices(
    class_count: int,
    scan_count: int,
    cfg: Mapping[str, Any],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """在每个气味内部按重复测量次数切分，避免训练/测试共享同一条扫描。"""

    split_cfg = dict(cfg.get("split") or {})
    ratios = [
        float(split_cfg.get("train", 0.70)),
        float(split_cfg.get("val", 0.15)),
        float(split_cfg.get("test", 0.15)),
    ]
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("spectra.split 的 train/val/test 必须非负且总和为 1。")
    train_n = int(round(scan_count * ratios[0]))
    val_n = int(round(scan_count * ratios[1]))
    test_n = scan_count - train_n - val_n
    if train_n < 1 or val_n < 1:
        raise ValueError("训练集和验证集每个气味至少需要 1 条扫描。")
    if test_n < 0:
        raise ValueError("光谱划分比例导致测试集长度为负。")

    strategy = str(split_cfg.get("strategy", "blocked")).lower()
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for class_index in range(class_count):
        base = class_index * scan_count
        local = list(range(scan_count))
        if strategy == "stratified":
            local = rng.permutation(scan_count).tolist()
        elif strategy != "blocked":
            raise ValueError("spectra.split.strategy 可选 blocked/stratified。")
        train_indices.extend(base + value for value in local[:train_n])
        val_indices.extend(base + value for value in local[train_n : train_n + val_n])
        test_indices.extend(base + value for value in local[train_n + val_n :])
    return train_indices, val_indices, test_indices


def build_spectral_datasets(
    config: Mapping[str, Any],
    seed: int,
    logger: Any = None,
) -> Any:
    """构建可供现有 Trainer 使用的 DataBundle。"""

    from .datasets import DataBundle

    cache = prepare_spectral_cache(config, logger=logger)
    dataset_cfg = config["dataset"]
    spectra_cfg = dict(dataset_cfg.get("spectra") or {})
    train_indices, val_indices, test_indices = _split_indices(
        len(cache.class_names),
        cache.scan_count,
        spectra_cfg,
        seed,
    )
    augment = spectra_cfg.get("augment") or {}
    train = SpectralArrayDataset(cache.x, cache.y, train_indices, augment, training=True)
    val = SpectralArrayDataset(cache.x, cache.y, val_indices, training=False)
    test = (
        SpectralArrayDataset(cache.x, cache.y, test_indices, training=False)
        if test_indices
        else None
    )
    return DataBundle(
        train=train,
        val=val,
        test=test,
        class_names=cache.class_names,
        input_shape=(1, len(cache.materials), cache.wavelengths.size),
        num_classes=len(cache.class_names),
        mean=(0.0,),
        std=(1.0,),
        train_size=len(train),
        val_size=len(val),
        test_size=len(test) if test is not None else 0,
        warnings=list(cache.metadata.get("warnings", [])),
    )


def make_odor_images(
    config: Mapping[str, Any],
    odor_dir: str | Path,
    logger: Any = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把一个新的气味目录处理成 [50,1,材料数,波长数] 预测张量。"""

    cache = prepare_spectral_cache(config, logger=logger)
    cfg = dict(config["dataset"].get("spectra") or {})
    dark_path = resolve_path(cfg.get("dark_avg"), PROJECT_ROOT)
    raw_dir = resolve_path(cfg.get("raw_dir"), PROJECT_ROOT)
    odor_path = resolve_path(odor_dir, PROJECT_ROOT)
    if dark_path is None or raw_dir is None or odor_path is None:
        raise ValueError("spectra.raw_dir/dark_avg 和 odor_dir 不能为空。")
    aliases = {str(key): str(value) for key, value in (cfg.get("material_aliases") or {}).items()}
    files = _material_files(odor_path, aliases)
    none_files = _material_files(raw_dir / "none", aliases)
    dark_w, dark_values = read_dark_avg(dark_path)
    target_w, target_dark = _select_wavelengths(
        dark_w,
        dark_values,
        cfg.get("wavelength_min"),
        cfg.get("wavelength_max"),
    )
    images = np.zeros(
        (cache.scan_count, 1, len(cache.materials), target_w.size),
        dtype=np.float32,
    )
    for material_index, material in enumerate(cache.materials):
        if material not in files:
            raise FileNotFoundError(f"预测目录 {odor_path} 缺少材料 {material} 的 txt。")
        if material not in none_files:
            raise FileNotFoundError(f"none 目录缺少材料 {material} 的 txt。")
        images[:, 0, material_index, :] = _process_spectrum_pair(
            files[material],
            none_files[material],
            target_dark,
            target_w,
            cfg,
        )
    return images, {
        "odor_dir": str(odor_path),
        "materials": cache.materials,
        "wavelengths": target_w.tolist(),
        "scan_count": cache.scan_count,
    }

