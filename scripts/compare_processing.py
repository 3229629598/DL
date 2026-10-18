"""比较不同归一化/空白相减方式的差谱幅度，辅助选择可视化参数。

只处理 2-辛酮一个气味，不会覆盖正式缓存。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnnlab.spectra import (
    _material_files,
    _process_spectrum_pair,
    _select_wavelengths,
    read_dark_avg,
)


RAW_ROOT = PROJECT_ROOT / "data" / "[original data]"


def main() -> None:
    aliases = {"UiO6Tb309": "UiO66Tb309", "UiO6Tb365": "UiO66Tb365"}
    none_files = _material_files(RAW_ROOT / "none", aliases)
    gas_files = _material_files(RAW_ROOT / "2-辛酮", aliases)
    dark_w, dark = read_dark_avg(RAW_ROOT / "dark_avg.xlsx")
    wavelengths, dark = _select_wavelengths(dark_w, dark, 400.411, 849.672)
    materials = [
        "808-365", "808Eu309", "808Eu365", "808Tb309", "808Tb365",
        "AlEu", "Albpydc", "EuBDC", "EuBDC-OH", "EuBTC", "EuNDC",
        "GaEu", "Gabpydc", "TbBDC", "TbBTC", "UiO66Eu309",
        "UiO66Eu365", "UiO66Tb309", "UiO66Tb365", "ZUM", "ZUMEu",
        "ZrEu", "Zrbpydc",
    ]
    variants = [
        ("wavelength/mean", {"normalize_axis": "wavelength", "blank_mode": "mean"}),
        ("wavelength/paired", {"normalize_axis": "wavelength", "blank_mode": "paired"}),
        ("scan/mean", {"normalize_axis": "scan", "blank_mode": "mean"}),
        ("none/mean", {"normalize_axis": "none", "blank_mode": "mean"}),
    ]
    for label, variant in variants:
        cfg = {
            "scan_count": 50,
            "smooth_window": 9,
            "normalize_axis": variant["normalize_axis"],
            "blank_mode": variant["blank_mode"],
        }
        stack = []
        for material in materials:
            stack.append(
                _process_spectrum_pair(
                    gas_files[material],
                    none_files[material],
                    dark,
                    wavelengths,
                    cfg,
                )
            )
        array = np.asarray(stack).mean(axis=1)
        print(
            f"{label:20s} min={array.min():7.3f} max={array.max():7.3f} "
            f"p1/p99={np.percentile(array, [1,99]).round(3)} "
            f"mean_abs={np.mean(np.abs(array)):.3f}"
        )


if __name__ == "__main__":
    main()

