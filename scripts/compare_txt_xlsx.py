"""核对 txt 原始扫描与 xlsx 导出表是否一致。

txt 是光谱仪原始导出，通常包含完整的 50 条扫描；
xlsx 是二次导出，个别文件会少几列。本脚本读取一条样本，
确认 xlsx 的波长轴和数值确实来自 txt，之后才能安全地用 txt 补齐短表。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "[original data]"


def parse_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """解析光谱仪 txt：返回 wavelengths[1044] 和 scans[50,1044]。"""

    text = path.read_bytes().decode("gbk", errors="replace")
    lines = text.splitlines()
    begin = next(index for index, line in enumerate(lines) if "Begin Spectral Data" in line)
    wavelengths = np.array([float(token) for token in lines[begin + 1].split() if token], dtype=np.float64)
    scans: list[list[float]] = []
    for line in lines[begin + 2 :]:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        # 前两列是时间戳和机器编号，其余才是光谱强度。
        values = []
        for token in tokens[2:]:
            try:
                values.append(float(token))
            except ValueError:
                # 跳过极少数夹带的非数值字段。
                continue
        if len(values) >= len(wavelengths):
            scans.append(values[: len(wavelengths)])
    if not scans:
        raise ValueError(f"没有从 {path} 解析到扫描数据。")
    return wavelengths, np.asarray(scans, dtype=np.float32)


def parse_xlsx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))
    workbook.close()
    wavelengths = np.array([float(row[0]) for row in rows if row[0] is not None], dtype=np.float64)
    scans = np.array(
        [[float(value) for value in row[1:] if value is not None] for row in rows if row[0] is not None],
        dtype=np.float32,
    )
    return wavelengths, scans


def main() -> None:
    material = "808-309"
    odor = "2-辛酮"
    txt_path = next((RAW_ROOT / odor).glob(f"{material}_*.txt"))
    xlsx_path = RAW_ROOT / odor / f"{material}.xlsx"
    txt_wavelengths, txt_scans = parse_txt(txt_path)
    xlsx_wavelengths, xlsx_scans = parse_xlsx(xlsx_path)

    print(f"txt：{txt_scans.shape}，波长 {txt_wavelengths[0]:.3f}..{txt_wavelengths[-1]:.3f}")
    print(f"xlsx：{xlsx_scans.shape}，波长 {xlsx_wavelengths[0]:.3f}..{xlsx_wavelengths[-1]:.3f}")

    indices = []
    for target in xlsx_wavelengths:
        indices.append(int(np.abs(txt_wavelengths - target).argmin()))
    indices_array = np.asarray(indices)
    print(f"波长最大匹配误差：{np.abs(txt_wavelengths[indices_array] - xlsx_wavelengths).max():.6f} nm")

    n_scans = min(xlsx_scans.shape[1], txt_scans.shape[0])
    txt_selected = txt_scans[:n_scans, :][:, indices_array]
    xlsx_selected = xlsx_scans[:, :n_scans].T
    print(f"比较扫描数：{n_scans}")
    print(f"数值最大绝对误差：{np.abs(txt_selected - xlsx_selected).max():.8f}")
    print(f"数值平均绝对误差：{np.abs(txt_selected - xlsx_selected).mean():.8f}")
    print(f"前 5 个 xlsx 值：{xlsx_selected[0, :5]}")
    print(f"前 5 个 txt 值：{txt_selected[0, :5]}")


if __name__ == "__main__":
    main()

