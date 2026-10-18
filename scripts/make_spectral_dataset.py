"""生成光谱 CNN 数据集和全部气味热图。

用法：
    python scripts/make_spectral_dataset.py --config configs/spectral_cnn.yaml
    python scripts/make_spectral_dataset.py --config configs/spectral_cnn.yaml --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnnlab.config import load_config
from cnnlab.spectra import prepare_spectral_cache
from cnnlab.utils import configure_matplotlib_cache, create_run_dir, resolve_path, setup_logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建光谱缓存和气味热图")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="忽略缓存，重新解析全部原始 txt")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    args = parse_args(argv)
    config = load_config(args.config)
    output_base = resolve_path("outputs/spectra", PROJECT_ROOT)
    assert output_base is not None
    run_dir = create_run_dir(output_base, prefix="dataset_build", unique=True)
    logger = setup_logger(run_dir, log_file="build.log")
    configure_matplotlib_cache(run_dir / "matplotlib_cache")
    cache = prepare_spectral_cache(config, logger=logger, force_rebuild=args.force)
    logger.info(
        "完成：%d 个样本，%d 个气味，%d 种材料，%d 个波长点。",
        cache.x.shape[0],
        len(cache.class_names),
        len(cache.materials),
        cache.wavelengths.size,
    )
    logger.info("缓存文件：%s", cache.cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

