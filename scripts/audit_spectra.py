"""扫描原始光谱 Excel，检查数据完整性和列对齐情况。

用法：
    python scripts/audit_spectra.py

该脚本只读，不会修改原始数据。它帮助确认：
- 每个气味目录包含哪些材料；
- 每个 Excel 的波长行数、测量列数；
- 各气味之间材料名称是否一致；
- 是否存在缺列、空值或波长轴不一致。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "[original data]"


def read_xlsx(path: Path) -> tuple[list[float], list[list[float]]]:
    """读取一个光谱 Excel。

    返回：
    - wavelengths：第一列波长；
    - columns：其余每一列是一条光谱测量。
    """

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.active
    wavelengths: list[float] = []
    columns: list[list[float]] = []
    for row_index, row in enumerate(sheet.iter_rows(values_only=True)):
        if row_index == 0:
            # 第一列是波长，后续每个非空单元格是一列测量。
            columns = [[] for _ in row[1:]]
        if row[0] is None:
            continue
        wavelengths.append(float(row[0]))
        for column_index, value in enumerate(row[1:]):
            if column_index >= len(columns):
                columns.append([])
            if value is None:
                columns[column_index].append(float("nan"))
            else:
                columns[column_index].append(float(value))
    workbook.close()
    return wavelengths, columns


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{RAW_ROOT}")

    odor_dirs = sorted(path for path in RAW_ROOT.iterdir() if path.is_dir())
    all_material_names: set[str] = set()
    column_counts: Counter[int] = Counter()
    row_counts: Counter[int] = Counter()
    wavelength_shapes: Counter[tuple[float, float, int]] = Counter()
    missing_values = 0
    total_files = 0
    per_odor: dict[str, dict[str, object]] = {}

    for odor_dir in odor_dirs:
        xlsx_files = sorted(odor_dir.glob("*.xlsx"))
        materials = sorted(path.stem for path in xlsx_files)
        all_material_names.update(materials)
        odor_column_counts: Counter[int] = Counter()
        odor_row_counts: Counter[int] = Counter()
        for path in xlsx_files:
            total_files += 1
            wavelengths, columns = read_xlsx(path)
            column_counts[len(columns)] += 1
            row_counts[len(wavelengths)] += 1
            odor_column_counts[len(columns)] += 1
            odor_row_counts[len(wavelengths)] += 1
            if wavelengths:
                wavelength_shapes[
                    (
                        round(wavelengths[0], 6),
                        round(wavelengths[-1], 6),
                        len(wavelengths),
                    )
                ] += 1
            missing_values += sum(
                1 for column in columns for value in column if value != value
            )
        per_odor[odor_dir.name] = {
            "material_count": len(materials),
            "materials": materials,
            "column_counts": dict(odor_column_counts),
            "row_counts": dict(odor_row_counts),
        }

    print(f"原始目录：{RAW_ROOT}")
    print(f"气味目录数：{len(odor_dirs)}")
    print(f"Excel 文件数：{total_files}")
    print(f"所有材料名（{len(all_material_names)}）：{sorted(all_material_names)}")
    print(f"列数分布：{dict(sorted(column_counts.items()))}")
    print(f"行数分布：{dict(sorted(row_counts.items()))}")
    print(f"波长轴分布（起点, 终点, 行数）：")
    for key, count in sorted(wavelength_shapes.items(), key=lambda item: item[0]):
        print(f"  {key}: {count} 个文件")
    print(f"缺失/NaN 数量：{missing_values}")

    inconsistent = {
        odor: info
        for odor, info in per_odor.items()
        if info["column_counts"] != {50: info["material_count"]}
    }
    print(f"列数不是 50 的气味数：{len(inconsistent)}")
    for odor, info in inconsistent.items():
        print(f"  {odor}: 材料 {info['material_count']}，列数 {info['column_counts']}")

    output = PROJECT_ROOT / "outputs" / "spectra_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "odor_count": len(odor_dirs),
                "file_count": total_files,
                "materials": sorted(all_material_names),
                "column_counts": dict(column_counts),
                "row_counts": dict(row_counts),
                "per_odor": per_odor,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"审计结果已保存：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

