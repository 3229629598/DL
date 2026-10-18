"""检查 xlsx 的 50 个测量列与 txt 的 50 条扫描如何对应。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from compare_txt_xlsx import parse_txt, parse_xlsx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data" / "[original data]"


def main() -> None:
    odor = "2-辛酮"
    material = "808-309"
    txt_path = next((RAW_ROOT / odor).glob(f"{material}_*.txt"))
    xlsx_path = RAW_ROOT / odor / f"{material}.xlsx"
    txt_w, txt_scans = parse_txt(txt_path)
    xlsx_w, xlsx_scans = parse_xlsx(xlsx_path)
    indices = np.array([int(np.abs(txt_w - target).argmin()) for target in xlsx_w])
    txt_selected = txt_scans[:, indices]

    errors = np.empty((xlsx_scans.shape[1], txt_scans.shape[0]))
    for column in range(xlsx_scans.shape[1]):
        errors[column] = np.mean(np.abs(txt_selected - xlsx_scans[:, column]), axis=1)
    best = errors.argmin(axis=1)
    print("每个 xlsx 列最接近的 txt 扫描索引：")
    print(best.tolist())
    print("对应平均绝对误差：")
    print(errors.min(axis=1).round(4).tolist())
    print("xlsx 第 0 列与 txt 各扫描误差（前 8）：")
    print(errors[0, :8].round(4).tolist())
    print("xlsx 最后一列与 txt 各扫描误差（后 8）：")
    print(errors[-1, -8:].round(4).tolist())


if __name__ == "__main__":
    main()

