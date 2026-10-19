# 光谱电子鼻气味识别项目

本项目基于 `cnn_test` 中的通用 CNN 框架，读取 `data/[original data]` 下的光谱原始数据，按“减暗电流 → z-score 归一化 → 减空白 → 平滑”的流程生成材料×波长差谱，并训练 CNN 判断气味类别。

初学者建议先阅读配套教程：[TUTORIAL.md](TUTORIAL.md)。它按“配置 → 原始数据 → 预处理 → CNN → 训练 → 评估 → 预测”的顺序解释整个项目。

## 数据情况

- 气味目录：51 个（`none` 是空白参考，不算气味类别）。
- 材料：所有气味和 none 的交集为 23 种。
- 每个材料的 txt：50 条扫描；每条扫描 1044 个原始波长点。
- 统一裁剪波长：400.411–849.672 nm，592 个波长点。
- 每个气味生成 50 个“重复扫描”样本，共 2550 个样本；输入张量为 `[1, 23, 592]`。

原始 xlsx 中有 56 个文件的列数少于 50（45–49 列），但对应 txt 均有完整 50 条扫描。因此数据管线以 txt 为光谱数值主来源，xlsx 只用于读取 `dark_avg.xlsx`，避免丢列或复制列。

审计脚本：

```powershell
python scripts/audit_spectra.py
```

## 处理流程

每条扫描执行：

1. `气体/空白原始强度 - dark_avg(波长)`；
2. 在波长方向做 z-score：`(x - mean_wavelength) / std_wavelength`；
3. 用 `none` 的 50 条扫描平均归一化光谱作为空白参考；
4. 气体归一化光谱 - 空白参考；
5. 沿波长方向做 9 点移动平均；
6. 同一气味的 50 条扫描生成 50 个样本，平均后得到一张热图。

空白相减支持：

- `blank_mode: mean`：减 none 的平均光谱，稳定、推荐；
- `blank_mode: paired`：第 j 条气体扫描减第 j 条 none 扫描。

归一化轴支持 `wavelength`、`scan`、`none`；当前默认 `wavelength`，即每条光谱在波长方向标准化。若业务上要求按 50 次测量做逐波长标准化，可修改 `spectra.normalize_axis: scan`。

## 快速开始

```powershell
conda activate dl1
cd G:\Users\3229629598\desktop\dl\cnn_test

# 生成光谱缓存、全部气味热图
python scripts/make_spectral_dataset.py --config configs/spectral_cnn.yaml

# 强制重新解析全部 txt 并重画热图
python scripts/make_spectral_dataset.py --config configs/spectral_cnn.yaml --force

# 训练光谱 CNN
python train.py --config configs/spectral_cnn.yaml

# 评估测试集
python evaluate.py --config configs/spectral_cnn.yaml `
  --checkpoint outputs/runs/spectra_20261018_225150_final/best.pth --split test

# 对一个新的气味目录预测
python predict_spectra.py --config configs/spectral_cnn.yaml `
  --checkpoint outputs/runs/spectra_20261018_225150_final/best.pth `
  --odor-dir "data/[original data]/2-辛酮" --topk 5
```

## 配置重点

主要配置在 `configs/spectral_cnn.yaml` 的 `dataset.spectra`：

| 参数 | 说明 |
|---|---|
| `raw_dir` | 原始气味数据根目录 |
| `dark_avg` | 暗电流 Excel |
| `cache_dir` | npz 缓存目录 |
| `scan_count` | 每个材料使用的扫描数，默认 50 |
| `wavelength_min/max` | 统一波长范围 |
| `material_strategy` | `intersection` 或 `union` |
| `material_aliases` | 修正文件名笔误 |
| `material_order` | 热图纵轴材料顺序 |
| `normalize_axis` | z-score 的方向 |
| `blank_mode` | `mean` 或 `paired` |
| `smooth_window` | 移动平均窗口 |
| `clip` | 可选差谱截断 |
| `split` | 训练/验证/测试比例和划分策略 |
| `augment` | 波长平移、缩放、噪声、掩码、材料丢弃 |
| `heatmaps` | 热图目录、色标、分辨率、总览图 |

训练配置沿用项目的通用 CNN 配置，支持优化器、学习率调度、AMP、梯度裁剪、梯度累积、早停、冻结层和断点续训。

## 输出文件

- `data/spectral_processed/spectral_dataset.npz`：2550 个光谱样本及标签；
- `data/spectral_processed/spectral_meta.json`：类别、材料、波长、处理参数和审计警告；
- `data/spectral_processed/sample_manifest.csv`：每个样本对应的气味类别和扫描编号；
- `outputs/spectra/heatmaps/<气味>.png`：每个气味一张材料×波长热图；
- `outputs/spectra/heatmaps/all_odors_heatmaps.png`：所有气味热图索引；
- `outputs/spectra/heatmaps/mean_diff_spectra.npz`：所有气味的平均差谱矩阵；
- `outputs/runs/spectra_*/`：日志、曲线、最佳/最后权重、测试指标、逐类和混淆矩阵结果。

## 划分方式与注意点

默认采用“每个气味内部按扫描顺序分块”：

- 前 35 条扫描训练；
- 中间 8 条验证；
- 最后 7 条测试。

这样测试集不会和训练集共享同一条扫描，且每个类别都出现在三个划分中。若使用 `stratified`，会在每个气味内部随机抽扫描，但同一气味的相邻扫描高度相关，指标可能偏乐观；正式报告时应说明划分策略。

本次最终配置在测试集上取得 100% 准确率（357 条测试扫描），这说明同一批次内的重复扫描可分性很强。若需要评估跨批次/跨时间的泛化能力，应再按采集日期或独立批次划分，而不能只按扫描随机划分。

## 当前结果

已生成的最终训练目录：

`outputs/runs/spectra_20261018_225150_final/`

- 验证集准确率：100%；
- 测试集准确率：100%，macro-F1：100%；
- 最佳权重：`best.pth`；
- 预测示例：`outputs/spectra/predict_2-辛酮.csv`，输入 2-辛酮目录，输出 2-辛酮置信度约 98.07%。

