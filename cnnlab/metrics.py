"""分类指标累计、混淆矩阵与可视化。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """逐元素除法；分母为 0 时返回 0，避免 NaN 污染指标。"""

    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
    """由混淆矩阵计算每类与宏平均/加权平均指标。

    混淆矩阵行是真值、列是预测值：
    - 精确率 precision = 对角线 / 该列预测总数
    - 召回率 recall    = 对角线 / 该行真实总数
    - F1              = 2 * precision * recall / (precision + recall)
    """

    matrix = confusion.float()
    true_positive = torch.diag(matrix)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    total = float(matrix.sum().item())
    weights = support / total if total > 0 else torch.zeros_like(support)

    return {
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.long().tolist(),
        "precision_macro": float(precision.mean().item()) if precision.numel() else 0.0,
        "recall_macro": float(recall.mean().item()) if recall.numel() else 0.0,
        "f1_macro": float(f1.mean().item()) if f1.numel() else 0.0,
        "precision_weighted": float((precision * weights).sum().item()) if total > 0 else 0.0,
        "recall_weighted": float((recall * weights).sum().item()) if total > 0 else 0.0,
        "f1_weighted": float((f1 * weights).sum().item()) if total > 0 else 0.0,
    }


@dataclass
class MetricAccumulator:
    """按 batch 累计损失、正确数和混淆矩阵。"""

    num_classes: int
    topk: tuple[int, ...] = (1, 5)
    binary_threshold: float = 0.5
    total_loss: float = 0.0
    total_samples: int = 0
    correct_at_k: dict[int, int] = field(default_factory=dict)
    confusion: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.topk = tuple(sorted({int(k) for k in self.topk if int(k) >= 1}))
        if not self.topk:
            self.topk = (1,)
        self.correct_at_k = {k: 0 for k in self.topk}
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
        )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        """累计一个 batch。

        logits 的两种合法形状：
        - [N,C]：多分类，C=num_classes。
        - [N,1]：二分类单输出，num_classes 必须为 1。
        """

        targets = targets.long()
        batch_size = targets.numel()
        if batch_size == 0:
            return
        self.total_loss += float(loss) * batch_size
        self.total_samples += batch_size

        if self.num_classes == 1:
            if logits.dim() != 2 or logits.shape[1] != 1:
                raise ValueError(f"二分类期望 logits 形状 [N,1]，实际为 {tuple(logits.shape)}。")
            prediction = (torch.sigmoid(logits[:, 0]) >= self.binary_threshold).long()
            # 二分类没有有意义的 top-k（k>1），只统计 top1 准确率。
            self.correct_at_k[1] = self.correct_at_k.get(1, 0) + int(
                (prediction == targets).sum().item()
            )
            self.correct_at_k = {key: value for key, value in self.correct_at_k.items() if key == 1}
        else:
            if logits.dim() != 2 or logits.shape[1] != self.num_classes:
                raise ValueError(
                    f"多分类期望 logits 形状 [N,{self.num_classes}]，实际为 {tuple(logits.shape)}。"
                )
            prediction = logits.argmax(dim=1)
            max_k = min(max(self.topk, default=1), self.num_classes)
            top_indices = logits.topk(k=max_k, dim=1).indices
            for k in self.topk:
                if k > self.num_classes:
                    # k 超过类别数时，其准确率与最大可用 k 相同。
                    effective_k = self.num_classes
                else:
                    effective_k = k
                hits = (top_indices[:, :effective_k] == targets[:, None]).any(dim=1)
                self.correct_at_k[k] += int(hits.sum().item())

        flat_index = targets.cpu() * self.num_classes + prediction.cpu()
        self.confusion += torch.bincount(
            flat_index,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        """返回当前累计结果；没有样本时返回全 0 指标。"""

        assert self.confusion is not None
        result: dict[str, Any] = {
            "loss": self.total_loss / self.total_samples if self.total_samples else 0.0,
            "num_samples": self.total_samples,
            "accuracy": (
                self.correct_at_k.get(1, 0) / self.total_samples if self.total_samples else 0.0
            ),
        }
        for k, correct in self.correct_at_k.items():
            result[f"top{k}_accuracy"] = correct / self.total_samples if self.total_samples else 0.0

        details = metrics_from_confusion(self.confusion)
        result.update(details)
        result["confusion_matrix"] = self.confusion.tolist()
        return result


def save_confusion_csv(
    confusion: torch.Tensor | Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: Path,
) -> None:
    """把混淆矩阵保存为带类别表头的 CSV（UTF-8 BOM 便于 Excel 打开）。"""

    matrix = torch.as_tensor(confusion).long()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix.tolist()):
            writer.writerow([name, *row])


def save_per_class_csv(metrics: dict[str, Any], class_names: Sequence[str], path: Path) -> None:
    """保存每类 precision/recall/F1/support，便于定位表现差的类别。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["class_index", "class_name", "precision", "recall", "f1", "support"])
        for index, name in enumerate(class_names):
            writer.writerow(
                [
                    index,
                    name,
                    metrics["precision_per_class"][index],
                    metrics["recall_per_class"][index],
                    metrics["f1_per_class"][index],
                    metrics["support_per_class"][index],
                ]
            )


def plot_confusion_matrix(
    confusion: torch.Tensor | Sequence[Sequence[int]],
    class_names: Sequence[str],
    path: Path,
    normalize: bool = True,
) -> None:
    """绘制混淆矩阵 PNG。

    类别很多时不叠加数字，避免图片变成不可读的黑块。
    图表标签使用英文，避免不同系统缺少中文字体而产生方框。
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = torch.as_tensor(confusion).float().cpu().numpy()
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)
    count = len(class_names)
    size = max(6.0, min(24.0, 0.42 * count + 3.0))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=range(count),
        yticks=range(count),
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix" + (" (row-normalized)" if normalize else ""),
    )
    axis.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8 if count > 20 else 9)
    axis.set_yticklabels(class_names, fontsize=8 if count > 20 else 9)

    if count <= 30:
        threshold = matrix.max() * 0.5 if matrix.size else 0.0
        for row in range(count):
            for column in range(count):
                value = matrix[row, column]
                text = f"{value:.2f}" if normalize else f"{int(value)}"
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)

