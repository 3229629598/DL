# 从零读懂这个光谱气味 CNN 项目

这是一篇面向深度学习初学者的项目导读。目标不是让你背下所有代码，而是让你能做到：

1. 知道项目从哪里读数据、数据怎样一步步变成 CNN 输入；
2. 看懂配置文件、模型结构、训练循环和指标文件；
3. 能自己修改归一化、网络层、学习率和数据划分；
4. 能解释当前 100% 准确率代表什么，以及它不代表什么。

建议边读边打开对应文件。教程中的路径都相对于：

```text
G:\Users\3229629598\desktop\dl\cnn_test
```

---

## 0. 先记住一句话

这个项目做的事情可以概括为：

```text
原始光谱 txt
  -> 减暗电流
  -> z-score
  -> 减空白光谱
  -> 平滑
  -> 每个气味得到 50 张“材料 × 波长”图
  -> CNN 输出 51 个气味类别中的一个
```

一次训练并不是把“一个气味”只喂给网络一次。一个气味有 50 条扫描，因此会产生 50 个样本；51 个气味 × 50 条扫描 = 2550 个样本。

最终保留的训练结果是：

```text
outputs/runs/spectra_20261018_225150_final/
```

测试集 357 条扫描，准确率 100%。这个结果说明“同一批采集数据内部的重复扫描”很容易区分，但不等价于跨日期、跨仪器、跨批次的泛化能力。后面会专门解释。

---

## 1. 项目目录应该怎样读

先看一张总表：

| 路径 | 作用 | 初学阶段要不要精读 |
|---|---|---|
| `configs/spectral_cnn.yaml` | 光谱项目的总配置 | **必须精读** |
| `data/[original data]/` | 原始气味、材料 txt 和 dark_avg | **必须理解结构** |
| `cnnlab/spectra.py` | 光谱解析、预处理、缓存、热图、光谱 Dataset | **必须精读** |
| `cnnlab/datasets.py` | 通用数据集与 DataLoader；把 `spectra` 分派给光谱模块 | 先看入口和 `DataBundle` |
| `cnnlab/model.py` | 把 `model.layers/head` 组装成 CNN | **必须精读** |
| `cnnlab/layers.py` | 可配置卷积、池化、激活、归一化、残差层 | 先看 `LayerBlock` 和 `build_layers` |
| `cnnlab/losses.py` | 损失函数和类别权重 | 看 `build_loss` |
| `cnnlab/metrics.py` | 准确率、top-k、F1、混淆矩阵 | 看 `MetricAccumulator` |
| `cnnlab/engine.py` | Trainer、优化器、训练循环、断点保存 | **必须精读** |
| `cnnlab/config.py` | YAML 继承、校验、命令行覆盖 | 看 `load_config` |
| `cnnlab/utils.py` | 随机种子、设备、日志、路径工具 | 按需查 |
| `train.py` | 训练入口 | **必须精读** |
| `evaluate.py` | 用断点评估 | 精读 |
| `predict_spectra.py` | 对新气味目录做光谱预测 | 精读 |
| `scripts/make_spectral_dataset.py` | 生成 npz 数据集和全部热图 | 精读 |
| `scripts/audit_spectra.py` | 检查原始数据完整性 | 建议运行一次 |
| `scripts/compare_*.py` | 数据格式/归一化诊断工具 | 了解即可 |
| `tests/test_spectral.py` | 光谱数据与模型集成测试 | 建议读 |
| `tests/test_smoke.py` | 通用 CNN 框架测试 | 了解即可 |
| `configs/mnist_lenet.yaml` 等 | 通用图像 CNN 示例 | 不是光谱主线，先跳过 |
| `predict.py` | 普通图片预测 | 不是光谱主线，先跳过 |

学习时不要从 `layers.py` 第一行开始硬读。正确顺序是：

```text
README -> spectral_cnn.yaml -> make_spectral_dataset.py
-> spectra.py 的 read/process/prepare -> build_spectral_datasets
-> model.py -> layers.py -> train.py -> engine.py
-> metrics/evaluate -> predict_spectra
```

这就是“从数据到结果”的阅读顺序。

---

## 2. 必要的基础概念

### 2.1 配置不是装饰，是项目的“控制面板”

这个项目没有把学习率、层数、批量大小写死在 Python 里，而是写在 YAML：

```yaml
dataset:
  batch_size: 64
model:
  layers:
    - type: conv2d
      out_channels: 32
train:
  epochs: 30
optimizer:
  lr: 0.001
```

YAML 用缩进表示层级。冒号左边是键，右边是值。列表用 `-` 表示。

配置文件有两层：

- `configs/default.yaml`：通用默认值；
- `configs/spectral_cnn.yaml`：光谱项目覆盖项。

加载顺序是：

```text
default.yaml <- spectral_cnn.yaml <- 命令行 --set
```

字典递归合并，列表整体替换。比如光谱配置里的 `model.layers` 会完整替换默认网络。

### 2.2 张量 shape 是读代码的钥匙

PyTorch 的图片张量通常是：

```text
[N, C, H, W]
```

- `N`：batch size，一次喂给网络多少样本；
- `C`：通道数，RGB 图是 3，灰度图是 1；
- `H`：高度；
- `W`：宽度。

本项目的“图像”不是普通照片，而是材料×波长热图：

```text
[N, 1, 23, 592]
```

其中：

- 1 是人为增加的单通道；
- 23 是材料；
- 592 是波长点。

所以“卷积核在图像上滑动”，在这里就是“同时观察相邻材料和相邻波长”。

### 2.3 Dataset、DataLoader 和 batch

`Dataset` 负责回答一个问题：

```python
dataset[i]  # 第 i 个样本是什么？
```

本项目的 `SpectralArrayDataset.__getitem__` 返回：

```python
x.shape == (1, 23, 592)
label    # 0..50 的气味编号
```

`DataLoader` 负责：

- 按 batch 取样本；
- 打乱训练集；
- 可选多进程读取；
- 把多个样本堆成 `[64, 1, 23, 592]`。

### 2.4 前向传播和反向传播

前向传播：

```python
logits = model(x)
loss = criterion(logits, y)
```

反向传播：

```python
loss.backward()
```

PyTorch 的自动求导会计算每个参数的梯度 `parameter.grad`。之后优化器：

```python
optimizer.step()
optimizer.zero_grad()
```

你不需要手写卷积的导数，但需要知道每一行代码在做什么。

---

## 3. 原始数据到底是什么

### 3.1 目录结构

```text
data/[original data]/
├─ 2-辛酮/
│  ├─ 808-309_QEP058691_23-19-49-755.txt
│  ├─ 808-309.xlsx
│  ├─ 808Eu309_QEP...txt
│  ├─ 808Eu309.xlsx
│  └─ ...
├─ a松油烯/
├─ 丁香酚/
├─ ...
└─ none/
   ├─ 808-309_QEP...txt
   └─ ...
```

每个气味目录下面是一个个材料。`none` 不是第 52 个气味，而是没有气味时的空白参考。

实际统计结果是：

- 52 个目录；
- 去掉 `none` 后，51 个气味类别；
- 所有气味和 `none` 共有的材料为 23 种；
- 每个材料约有 50 条扫描。

### 3.2 txt 文件的内部结构

打开任意一个 txt，可以看到：

- 文件头包含日期、仪器、积分时间等；
- `Number of Pixels in Spectrum: 1044`；
- `Begin Spectral Data`；
- 下一行是 1044 个波长；
- 再往下是 50 行扫描数据；
- 每行前两列是时间戳和机器编号，后面才是强度。

简化成数组：

```text
wavelengths: [1044]
scans:       [50, 1044]
```

`read_scan_txt()` 的任务就是跳过文件头，找到波长行，再把 50 行数值解析成 NumPy 数组。

### 3.3 xlsx 和 txt 的关系

大多数 xlsx 是：

```text
592 行 × (1 个波长列 + 50 个测量列)
```

但审计发现 56 个 xlsx 只有 45–49 个测量列，而对应 txt 仍然是完整 50 条扫描。

所以本项目的策略是：

- 光谱数值以 txt 为主；
- xlsx 只用来读取 `dark_avg.xlsx`；
- txt 的 1044 个点插值到 dark_avg 的 592 个统一波长点；
- 不复制缺失列，也不丢弃整个材料。

运行审计：

```powershell
python scripts\audit_spectra.py
```

它会输出气味数、Excel 数、列数分布、行数分布和异常气味。

### 3.4 dark_avg 是什么

`dark_avg.xlsx` 是每个波长上的暗电流：

```text
波长 400.411 -> 14.2375...
波长 401.189 -> 15.1838...
...
```

测量得到的气体光谱包含仪器暗电流，因此要先：

```text
气体净信号 = 气体原始强度 - dark_avg
空白净信号 = 空白原始强度 - dark_avg
```

注意：暗电流是按波长逐点相减，而不是减一个固定常数。

---

## 4. 从一条光谱到一张热图

主线代码在 `cnnlab/spectra.py`。

### 4.1 第一步：裁剪波长

原始 txt 的 1044 个点覆盖 336–1117 nm，但项目统一使用：

```yaml
wavelength_min: 400.411
wavelength_max: 849.672
```

裁剪和插值后得到：

```text
每条扫描 [592]
每个材料 50 条扫描 [50, 592]
```

为什么必须统一？因为 CNN 的输入形状必须固定为 `[1, 23, 592]`，不能一个样本 580 点、另一个 610 点。

### 4.2 第二步：减暗电流

用 NumPy 表示：

```python
gas = gas - dark[None, :]       # [50,592] - [1,592]
blank = blank - dark[None, :]
```

这里的 `[None, :]` 把 `dark` 从 `[592]` 变成 `[1,592]`，广播到 50 行。

### 4.3 第三步：z-score

项目默认对每条扫描在波长方向做 z-score：

```text
z(λ) = (x(λ) - mean(x)) / std(x)
```

代码是 `_zscore(gas, axis=1)`。`axis=1` 表示沿着 592 个波长求均值和标准差，而不是沿着 50 条扫描。

计算后，每条光谱的均值为 0、标准差约为 1。标准差为 0 时用 `1e-8` 防止除零。

配置项：

```yaml
normalize_axis: wavelength
```

可改为：

- `scan`：在每个波长上对 50 条扫描做 z-score；
- `none`：不归一化。

对当前数据，`scan` 会让每个波长上的 50 条扫描均值为 0，之后相减很容易把有用差异抹掉；所以默认是 `wavelength`。

### 4.4 第四步：减空白

项目默认用 `none` 的 50 条扫描平均光谱作为空白参考：

```python
blank_reference = blank.mean(axis=0)
difference = gas - blank_reference
```

也可以配置成一一配对：

```yaml
blank_mode: paired
```

这时是第 j 条气体扫描减第 j 条空白扫描。

为什么默认用平均空白？因为空白测量和气体测量不一定能一一对应到同一个时间点；平均空白更稳定。

### 4.5 第五步：平滑

差谱再沿波长方向做 9 点移动平均：

```yaml
smooth_window: 9
```

窗口越大，曲线越平滑，但可能模糊窄峰。偶数会自动变成下一个奇数。

### 4.6 第六步：形成样本

对一个气味、一个材料：

```text
50 条气体扫描
-> 50 条差谱
-> 每条 [592]
-> 材料矩阵 [50, 23, 592]
```

对于 51 个气味：

```text
X.shape = [51*50, 1, 23, 592]
        = [2550, 1, 23, 592]
y.shape = [2550]
```

`X` 是模型输入，`y` 是气味标签。

同一气味 50 条扫描平均：

```text
mean_diff[气味] = [23, 592]
```

这就是热图的来源。

### 4.7 热图为什么有几个大色带？

热图每一行是一种材料，每一列是一个波长：

- 红色：该材料在该波长上比空白强；
- 蓝色：比空白弱；
- 白色：接近空白。

图上的色带代表材料-波长组合对某个气味的响应模式。CNN 学的就是这些局部纹理和色带组合。

运行：

```powershell
python scripts\make_spectral_dataset.py --config configs\spectral_cnn.yaml
```

会生成：

```text
outputs/spectra/heatmaps/<气味>.png
outputs/spectra/heatmaps/all_odors_heatmaps.png
outputs/spectra/heatmaps/mean_diff_spectra.npz
```

---

## 5. 数据集怎么切分，为什么不能随便随机

### 5.1 本项目的 2550 个样本

```text
51 个气味 × 50 条扫描 = 2550
```

默认划分比例：

```yaml
split:
  train: 0.70
  val: 0.15
  test: 0.15
  strategy: blocked
```

每个气味：

```text
35 条训练 + 8 条验证 + 7 条测试
```

总计：

```text
train = 51 * 35 = 1785
val   = 51 * 8  = 408
test  = 51 * 7  = 357
```

### 5.2 blocked 和 stratified

`blocked`：按扫描顺序分块，前 35 条训练，中间 8 条验证，最后 7 条测试。

`stratified`：在每个气味内部随机抽扫描。

为什么默认 blocked？因为同一次采集的相邻扫描高度相似。如果随机抽，训练集和测试集可能包含几乎相同的相邻扫描，指标会虚高。blocked 至少让测试扫描在时间顺序上和训练分开。

即便如此，50 条扫描仍然来自同一批采集环境。真正的跨批次评估应该按日期、仪器状态或独立实验批次划分，而不是只按扫描编号。

### 5.3 数据增强

`SpectralArrayDataset` 在训练时才做增强：

```yaml
augment:
  random_shift: 4
  scale: [0.95, 1.05]
  noise_std: 0.02
  wavelength_mask_p: 0.0
  material_dropout_p: 0.0
```

含义：

- `random_shift`：波长方向平移最多 4 点；
- `scale`：整体乘以一个随机系数；
- `noise_std`：加入高斯噪声；
- `wavelength_mask`：随机遮掉一段波长；
- `material_dropout`：随机把某些材料行置零。

验证和测试不做这些随机增强，否则指标不可比较。

---

## 6. CNN 结构逐层读

打开 `configs/spectral_cnn.yaml` 的 `model.layers`。

### 6.1 输入形状

```text
[N, 1, 23, 592]
```

这里的 `N` 是 batch，后面三个维度才是单张热图的形状。

### 6.2 第一段卷积

```yaml
- type: conv2d
  out_channels: 32
  kernel_size: 3
  stride: 1
  padding: same
  norm: {type: batchnorm2d}
  activation: {name: relu, inplace: true}
```

含义：

- 32 个卷积核；
- 每个核 3×3；
- `same` 保持高度和宽度；
- BatchNorm 归一化每个通道；
- ReLU 把负值变成 0。

形状：

```text
[N,1,23,592] -> [N,32,23,592]
```

参数量（bias 被 BatchNorm 取代）：

```text
权重 32×1×3×3 = 288
BatchNorm 权重/偏置 2×32 = 64
合计 352
```

这与 `model_summary.txt` 中的 `352` 对得上。

### 6.3 池化

```yaml
- type: maxpool2d
  kernel_size: 2
  stride: 2
```

每次在 2×2 区域取最大值，把高宽减半：

```text
[N,32,23,592] -> [N,32,11,296]
```

池化没有可学习参数。它减少计算量，并让后续卷积看到更大的感受野。

### 6.4 完整形状追踪

| 层 | 输出 shape（不含 batch） | 说明 |
|---|---|---|
| 输入 | `[1,23,592]` | 材料×波长 |
| conv 32 | `[32,23,592]` | 3×3 same |
| maxpool 2 | `[32,11,296]` | 高宽减半 |
| conv 64 | `[64,11,296]` | 3×3 same |
| maxpool 2 | `[64,5,148]` | 高宽减半 |
| conv 128 | `[128,5,148]` | 3×3 same |
| maxpool 2 | `[128,2,74]` | 高宽减半 |
| conv 128 | `[128,2,74]` | 3×3 same |
| adaptive avgpool 1 | `[128,1,1]` | 每个通道取全局平均 |
| flatten | `[128]` | 变成向量 |
| dropout | `[128]` | 训练时随机丢弃 |
| linear | `[51]` | 51 个气味 logits |

把这条表背下来，比死记代码更有用。以后你改网络时，先问：

> 这一层把 shape 从什么变成什么？

### 6.5 为什么最后是 51

`head` 的最后一个线性层写的是：

```yaml
out_features: num_classes
```

`num_classes` 由数据自动推断为 51。因此输出是 51 个未归一化的分数，叫 logits。softmax 之后才是概率。

### 6.6 配置层是如何变成 PyTorch 模块的

核心在 `cnnlab/layers.py`：

- `_build_core()`：创建卷积、池化、线性等核心模块；
- `_build_norm()`：创建 BatchNorm/GroupNorm/LayerNorm；
- `_build_activation()`：创建 ReLU/GELU/SiLU 等；
- `_build_dropout()`：创建 Dropout；
- `LayerBlock`：按 `core -> norm -> activation -> dropout` 的顺序执行；
- `build_layers()`：逐层构建并自动传递 shape；
- `_conv_output_shape()`：根据卷积公式推导输出高宽。

`model.py` 再把 `features` 和 `head` 包成 `ConfigurableCNN`：

```python
features = nn.Sequential(...)
head = nn.Sequential(...)
```

### 6.7 为什么 BatchNorm 后面常常不写 bias

卷积的 bias 会被 BatchNorm 的平移抵消一部分，所以配置里 `bias: null` 会在有归一化时自动不加 bias，减少冗余参数。

### 6.8 Dropout 在做什么

`p=0.35` 表示训练时每个神经元有 35% 概率被临时置零。它让网络不要过度依赖某几个特征，起到正则化作用。验证和测试时 Dropout 自动关闭。

---

## 7. 损失、优化器和反向传播

### 7.1 logits 和概率

模型最后输出：

```text
logits.shape = [N, 51]
```

logits 是未归一化分数，不是概率。交叉熵损失内部会对 logits 做 log-softmax。预测时才会做：

```python
probabilities = torch.softmax(logits, dim=1)
```

### 7.2 交叉熵损失

配置：

```yaml
loss:
  name: cross_entropy
  params:
    weight: null
    label_smoothing: 0.05
```

交叉熵衡量“模型给正确类别的概率有多高”。越接近 1，损失越小。

`label_smoothing: 0.05` 表示不要要求模型把概率做到绝对的 1 和 0，而是给其他类别留一点概率，通常能减轻过拟合。

`weight: null` 表示每个类别权重相同。数据中每个气味都有 50 条扫描，基本均衡，所以暂时不需要类别加权。

### 7.3 优化器 AdamW

```yaml
optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.01
```

`lr` 是学习率，控制每次参数更新的步子。`weight_decay` 是权重衰减，抑制参数变得过大。

`no_weight_decay_for_bias_and_norm: true` 会让 bias 和 BatchNorm 参数不做 weight decay，这是常见做法。

`engine.py` 的 `build_optimizer()` 会把参数分成不同的组。你可以打开 `resolved_config.yaml` 或训练日志确认实际使用的优化器配置。

### 7.4 学习率调度器

```yaml
scheduler:
  name: cosine
  step_on: epoch
```

`cosine` 会让学习率按余弦曲线逐渐降低。学习率太低会学得慢，太高会导致 loss 震荡或变成 NaN。

### 7.5 混合精度 AMP

```yaml
amp:
  enabled: true
  dtype: bfloat16
```

混合精度用较低精度计算部分运算，减少显存占用、提高速度。当前使用 `bfloat16`，不需要像 `float16` 那样频繁使用 GradScaler。

代码会在训练器初始化时检查设备：如果不是 CUDA，会自动关闭 AMP。

### 7.6 梯度裁剪

```yaml
backward:
  grad_clip:
    enabled: true
    mode: norm
    max_norm: 1.0
```

如果梯度向量的范数超过 1.0，就把它按比例缩小。它可以防止个别异常 batch 导致参数爆炸。

### 7.7 梯度累积

```yaml
grad_accumulation_steps: 1
```

如果显存不够，可以把它改成 2 或 4：

- 前几个 batch 只累积梯度，不更新参数；
- 到指定步数才执行一次 `optimizer.step()`。

这相当于用更小的显存模拟更大的 batch。

### 7.8 早停

```yaml
early_stopping:
  enabled: true
  patience: 8
  restore_best_weights: true
```

如果验证集准确率连续 8 个 epoch 没有提升，就停止训练，并恢复验证集最好的权重。最终这次训练在第 16 个 epoch 早停，但保存的是第 8 个 epoch 的最佳权重。

---

## 8. 训练入口和训练循环

### 8.1 train.py 的流程

打开 `train.py`，从上往下可以读到：

```text
parse_args
  -> load_config
  -> set_seed
  -> create_run_dir
  -> setup_logger
  -> build_datasets
  -> build_model
  -> build_dataloaders
  -> Trainer(...)
  -> trainer.fit()
```

每一行的含义：

- `load_config`：合并 YAML 和 `--set`；
- `set_seed`：固定 Python/NumPy/PyTorch 随机种子；
- `create_run_dir`：创建本次训练输出目录；
- `build_datasets`：光谱数据会自动分派到 `cnnlab/spectra.py`；
- `build_model`：读取 `model.layers/head` 动态建网络；
- `build_dataloaders`：把 Dataset 包装成 batch；
- `Trainer`：把模型、数据、损失、优化器、日志组织起来；
- `fit`：执行 epoch 循环。

### 8.2 Trainer 初始化做了什么

`cnnlab/engine.py` 里的 `Trainer.__init__` 会：

1. 读取训练配置；
2. 判断 AMP 是否可用；
3. 读取训练标签并构建损失函数；
4. 构建优化器；
5. 构建学习率调度器；
6. 可选构建 GradScaler；
7. 读取断点（如果配置了 `resume_from`）；
8. 保存类别映射和数据集元信息。

`build_loss`、`build_optimizer` 和 `build_scheduler` 都尽量把“配置”与“PyTorch 对象”分开。这样你只改 YAML，不必改训练代码。

### 8.3 一个 epoch 的真实顺序

简化后的训练循环：

```python
model.train()
optimizer.zero_grad()
for inputs, targets in train_loader:
    with autocast():
        logits = model(inputs)
        loss = criterion(logits, targets)
    loss.backward()
    clip_gradients()
    optimizer.step()
    optimizer.zero_grad()
```

真实代码还处理了：

- 梯度累积；
- AMP 的 unscale；
- NaN 检测；
- 优化器步数；
- batch 日志；
- 学习率展示。

### 8.4 一个 epoch 后做什么

```text
训练一个 epoch
  -> 验证集评估
  -> 根据 monitor 判断是否刷新最佳
  -> 保存 best.pth
  -> 保存 last.pth
  -> 更新 history.csv
  -> 判断是否早停
```

`monitor: val_accuracy` 表示以验证准确率作为选最佳权重的指标。也可以改成 `val_loss`、`val_f1_macro`。

### 8.5 断点里有什么

`best.pth` / `last.pth` 不是只有模型参数，`_save_checkpoint()` 还保存：

- `model_state`：模型权重；
- `optimizer_state`：优化器状态；
- `scheduler_state`：学习率调度状态；
- `scaler_state`：AMP 状态；
- `config`：训练配置；
- `class_names`：类别顺序；
- `input_shape`：模型输入形状；
- `history`：训练历史；
- `metrics`：验证/测试指标。

所以 `evaluate.py` 和 `predict_spectra.py` 可以从断点恢复类别名称和输入尺寸，而不用重新猜。

### 8.6 训练日志怎么读

典型日志：

```text
Epoch 8/30 | batch 28/28 | loss 0.68 | acc 95% | lr 8.3e-4
Epoch 8/30 完成 | train_loss 0.68 | val_loss 0.54 | val_acc 100%
发现更优 val_accuracy=1.000000，已保存 best.pth
```

- batch loss：当前 batch 的损失；
- acc：当前 batch 准确率；
- train_loss：整个 epoch 训练平均损失；
- val_loss：验证集损失；
- val_acc：验证集准确率；
- lr：当前学习率。

日志里的 `best` 变大或 `loss` 下降，通常说明模型在学习。

---

## 9. 指标不是只看 accuracy

### 9.1 top-k 准确率

`top1` 是预测概率最高的类别是否正确。

`top5` 是正确类别是否在前 5 个预测中。即使 top1 错了，top5 仍可能正确。对于类别多的任务，top5 能帮助判断“模型是不是只差一点”。

### 9.2 混淆矩阵

`outputs/runs/.../test_confusion.png` 是混淆矩阵：

- 行：真实气味；
- 列：预测气味；
- 对角线：预测正确；
- 非对角线：混淆。

`metrics.py` 的 `MetricAccumulator` 在验证/测试时累计混淆矩阵，最后计算：

- precision：预测为某类中有多少是真的；
- recall：某类真实样本中有多少被找回；
- F1：precision 和 recall 的调和平均；
- macro-F1：每个类别 F1 的平均，不按样本数加权。

### 9.3 逐类指标

`test_per_class.csv` 会列出每个气味的 precision、recall、F1 和 support。即使总体 100%，也建议看逐类文件，确认没有某个类别样本数异常。

### 9.4 当前 100% 应该怎样理解

当前测试集来自同一次采集批次的不同扫描。模型可能学到了：

- 材料通道的稳定响应模式；
- 某个波长区间的系统差异；
- 同一气味重复扫描之间的高度一致性。

这证明数据管线、模型和训练循环是正确的，也说明在当前批次内识别很容易；但它不是“换一个实验室、换一天、换一个仪器仍然 100%”的证明。

---

## 10. 评估和预测入口

### 10.1 evaluate.py

```powershell
python evaluate.py --config configs\spectral_cnn.yaml `
  --checkpoint outputs\runs\spectra_20261018_225150_final\best.pth `
  --split test
```

它做的是：

1. 读取配置；
2. 读取断点；
3. 检查断点输入形状是否与当前数据一致；
4. 重新构建模型并加载权重；
5. 在验证集或测试集上累计指标；
6. 保存 JSON、逐类 CSV 和混淆矩阵。

它不会重新训练模型。

### 10.2 predict_spectra.py

```powershell
python predict_spectra.py --config configs\spectral_cnn.yaml `
  --checkpoint outputs\runs\spectra_20261018_225150_final\best.pth `
  --odor-dir "data/[original data]/2-辛酮" `
  --topk 5
```

对一个新的气味目录，程序：

1. 找到该目录下每个材料的 txt；
2. 从 `raw_dir/none` 找到同名空白；
3. 重复训练时的减暗电流、z-score、减空白、平滑；
4. 生成 50 张 `[1,23,592]` 图；
5. 分别预测，得到 50 组 softmax 概率；
6. 对 50 组概率取平均；
7. 输出 top-k 类别。

所以它不是只拿一条扫描下结论，而是融合了 50 条扫描。

当前 2-辛酮预测结果是 98.07% 的 2-辛酮，输出在：

```text
outputs/spectra/predict_2-辛酮.csv
```

### 10.3 predict.py 和 predict_spectra.py 的区别

- `predict.py`：普通图片分类，使用 ImageFolder/图片 transform；
- `predict_spectra.py`：光谱气味分类，使用 txt 解析和 none 空白相减。

做光谱任务时不要误用 `predict.py`。

---

## 11. 配置文件逐段解释

`configs/spectral_cnn.yaml` 是光谱项目的核心控制面板。

### 11.1 dataset 段

```yaml
dataset:
  name: spectra
  root: data
  num_classes: null
  in_channels: 1
  image_size: [23, 592]
  batch_size: 64
  num_workers: 0
```

- `name: spectra` 是分派开关，告诉 `datasets.py` 使用光谱模块；
- `num_classes: null` 表示类别数从数据自动推断；
- `in_channels: 1` 是热图单通道；
- `image_size: [23,592]` 只是提示，真正的输入形状来自缓存；
- `batch_size` 是每批样本数；
- `num_workers` 是读取子进程数，Windows 初学者建议先用 0。

### 11.2 spectra 段

最重要的处理参数：

```yaml
scan_count: 50
wavelength_min: 400.411
wavelength_max: 849.672
material_strategy: intersection
normalize_axis: wavelength
blank_mode: mean
smooth_window: 9
```

`material_strategy: intersection` 表示只保留所有气味都有的 23 种材料。改成 `union` 会保留 26 种材料，缺失材料填零；这会把“材料是否存在”也变成模型输入，未必是好事。

`material_aliases` 修正了 `UiO6Tb...` 到 `UiO66Tb...` 的笔误。如果不修正，某些气味会因为材料名不一致而缺行。

`material_order` 控制热图纵轴顺序。这里按参考图排列，便于人工对照。

### 11.3 split 段

```yaml
split:
  train: 0.70
  val: 0.15
  test: 0.15
  strategy: blocked
```

三个比例必须加起来为 1。`blocked` 按顺序切分，`stratified` 随机切分。

### 11.4 augment 段

增强只影响训练集。想先验证原始性能，可以把所有增强设为 0：

```yaml
augment:
  random_shift: 0
  scale: [1.0, 1.0]
  noise_std: 0.0
```

### 11.5 heatmaps 段

```yaml
heatmaps:
  enabled: true
  output_dir: outputs/spectra/heatmaps
  cmap: RdBu_r
  vmin: -4.0
  vmax: 4.0
  dpi: 200
```

`RdBu_r` 是蓝-白-红配色，适合表示正值/负值。`vmin/vmax` 固定色标范围，避免每个气味的色标不同导致无法比较。

### 11.6 model 段

`model.layers` 是特征提取器，`model.head` 是分类头。可以在 `layers` 中加入：

```yaml
- type: conv2d
  out_channels: 64
  kernel_size: 3
  padding: same
  norm: {type: batchnorm2d}
  activation: {name: gelu}
```

也可以把 `relu` 改成 `leaky_relu`、`silu`、`gelu`、`mish` 等。所有层的输入形状由 `build_layers` 自动传递，不需要手填输入通道数。

### 11.7 train、loss、optimizer、scheduler

这些段控制训练策略：

- `train.epochs`：最多训练多少轮；
- `train.monitor`：按什么指标选最佳；
- `train.amp`：混合精度；
- `train.backward.grad_clip`：梯度裁剪；
- `loss`：损失函数和 label smoothing；
- `optimizer`：AdamW、学习率、权重衰减；
- `scheduler`：余弦退火。

初学者先只改 `train.epochs` 和 `optimizer.lr`，不要一次改十几个参数。

---

## 12. 其余文件怎么看

### 12.1 cnnlab/config.py

负责：

- 读取 YAML；
- 递归合并默认配置；
- 处理 `--set`；
- 校验关键字段；
- 保存最终配置快照。

每次训练目录都有一份 `resolved_config.yaml`，这是最值得看的配置，因为它把默认值和覆盖值合并后的最终结果完整保存了。

### 12.2 cnnlab/utils.py

常用工具：

- `set_seed`：固定随机种子；
- `resolve_device`：选择 CPU/CUDA；
- `setup_logger`：同时写控制台和日志文件；
- `create_run_dir`：生成不重复输出目录；
- `save_json`、`load_json`；
- `describe_environment`：记录 Python/PyTorch/CUDA 版本。

### 12.3 cnnlab/layers.py

这是“层工厂”。配置里的 `type: conv2d` 会进入 `_build_core()`，配置里的 `norm` 会进入 `_build_norm()`，以此类推。

除了普通卷积，它还支持：

- `linear`、`flatten`；
- `maxpool2d`、`avgpool2d`、`adaptive_avgpool2d`；
- `global_avgpool`；
- `residual` 残差块；
- `sequence` 子层序列；
- `batchnorm`、`layernorm`、`groupnorm`；
- `dropout`、`dropout2d`；
- 多种激活函数；
- `trainable` 冻结层和 `checkpoint` 梯度检查点。

光谱项目本身只用了其中一小部分，但通用框架保留了这些能力。

### 12.4 cnnlab/datasets.py

它包含 MNIST/CIFAR/ImageFolder/合成图片等通用数据逻辑。最开始看到这些代码容易迷路，你只需要先看 `build_datasets()` 开头的分派：

```python
if name == "spectra":
    return build_spectral_datasets(...)
```

这说明光谱项目不会走图像 transform 分支。

### 12.5 cnnlab/losses.py 和 metrics.py

`losses.py` 负责把 `loss.name` 变成 PyTorch 损失函数；`metrics.py` 负责累计损失、准确率、top-k、混淆矩阵、逐类 F1。

### 12.6 tests

```powershell
python -m unittest discover -s tests -v
```

`test_spectral.py` 检查：

- txt 是 50 条扫描；
- dark_avg 有波长和暗电流；
- 数据 bundle 形状是 `[1,23,592]`；
- 51 类；
- 模型能前向得到 `[2,51]`。

测试通过说明“数据形状和模型接口没有坏”，但不代表模型一定泛化好。

---

## 13. 推荐的动手练习

### 练习 1：检查一条 txt

```powershell
python -c "from pathlib import Path; from cnnlab.spectra import read_scan_txt; p=next(Path('data/[original data]/2-辛酮').glob('808-309_*.txt')); w,s=read_scan_txt(p); print(w.shape, s.shape, w[0], w[-1])"
```

你应该看到类似：

```text
(1044,) (50,1044) 336.485 1117.936
```

### 练习 2：看缓存的 shape

```powershell
python -c "import numpy as np; d=np.load('data/spectral_processed/spectral_dataset.npz'); print(d['x'].shape, d['y'].shape)"
```

预期：

```text
(2550, 1, 23, 592) (2550,)
```

### 练习 3：改变平滑窗口

把 `smooth_window` 改成 1，运行：

```powershell
python scripts\make_spectral_dataset.py --config configs\spectral_cnn.yaml --force
```

观察热图是否更尖锐、噪声更多。再改回 9。

### 练习 4：改变空白相减方式

把：

```yaml
blank_mode: mean
```

改成：

```yaml
blank_mode: paired
```

重新生成并比较热图。注意：平均差谱可能近似相同，但单条样本的对应关系会改变。

### 练习 5：改变一个网络超参数

把第一层 `out_channels: 32` 改成 16：

```powershell
python scripts\make_spectral_dataset.py --config configs\spectral_cnn.yaml
python train.py --config configs\spectral_cnn.yaml --set train.epochs=2
```

观察 `model_summary.txt` 中的参数量变化。

### 练习 6：故意制造 shape 错误

把 `model.layers` 最后一个 `adaptive_avgpool2d` 删掉，运行 dry-run，阅读错误信息。重新加回来后应该恢复正常。

### 练习 7：训练短实验

```powershell
python train.py --config configs\spectral_cnn.yaml --set train.epochs=2
```

打开输出目录里的：

- `curves.png`：损失/准确率/F1/学习率；
- `history.csv`：每个 epoch 的数字；
- `model_summary.txt`：每层形状；
- `resolved_config.yaml`：实际生效配置。

---

## 14. 常见报错和排查

### 14.1 `ModuleNotFoundError`

确认你使用 `dl1`：

```powershell
conda activate dl1
python -c "import torch, torchvision, openpyxl, yaml; print(torch.__version__)"
```

如果 `conda activate` 不可用，直接使用：

```powershell
& 'C:\Users\3229629598\miniconda3\envs\dl1\python.exe' train.py --config configs\spectral_cnn.yaml
```

### 14.2 找不到 `data/[original data]`

文件夹名包含方括号，在 PowerShell 中要加引号：

```powershell
--odor-dir "data/[original data]/2-辛酮"
```

### 14.3 材料交集为空

通常是文件名不符合 `<材料>_QEP...txt`，或者某个目录多了/少了材料。运行：

```powershell
python scripts\audit_spectra.py
```

检查 `outputs/spectra_audit.json` 中的材料列表和 `material_aliases`。

### 14.4 缓存没有更新

修改了原始数据或预处理参数后：

```powershell
python scripts\make_spectral_dataset.py --config configs\spectral_cnn.yaml --force
```

`spectra.py` 会用原始文件大小、修改时间和配置生成指纹；正常情况下会自动判断缓存过期。

### 14.5 CUDA 显存不够

依次尝试：

1. 把 `dataset.batch_size` 从 64 改成 32；
2. 保持 `amp.enabled: true`；
3. 减少卷积通道数；
4. 减少波长点或材料数；
5. 确认没有同时运行其他占显存程序。

### 14.6 loss 变成 NaN

- 降低 `optimizer.lr`；
- 检查原始数据是否有 NaN；
- 确认 `grad_clip.enabled: true`；
- 不要一开始使用过大的学习率。

### 14.7 Windows DataLoader 卡住

先把：

```yaml
num_workers: 0
```

等整个流程跑通后再尝试 2 或 4。

### 14.8 字体缓存权限警告

项目会把 Matplotlib 缓存放到数据目录。若仍出现字体缓存警告，可以设置：

```powershell
$env:MPLCONFIGDIR = "G:\Users\3229629598\desktop\dl\cnn_test\data\spectral_processed\matplotlib_cache"
```

### 14.9 Git 报 `dubious ownership`

这是 Git 的所有者检查，不是代码错误。可以在命令前加：

```powershell
git -c safe.directory='G:/Users/3229629598/desktop/dl/cnn_test' -C cnn_test status
```

---

## 15. 术语表

| 术语 | 本项目里的意思 |
|---|---|
| batch | 一次送进模型的 64 个样本 |
| epoch | 把所有训练样本完整过一遍 |
| logits | 51 个未归一化类别分数 |
| loss | 模型预测与真实标签的误差 |
| backward | 用链式法则计算梯度 |
| optimizer | 根据梯度更新参数 |
| scheduler | 按训练进度调整学习率 |
| AMP | 混合精度训练 |
| BatchNorm | 对每个通道做归一化，稳定训练 |
| Dropout | 训练时随机屏蔽神经元 |
| overfitting | 训练集好、新数据差 |
| confusion matrix | 真实类别与预测类别的交叉计数 |
| macro-F1 | 每个类别 F1 的简单平均 |
| data leakage | 测试信息以某种方式泄漏到训练中 |
| cache | 预处理结果的 npz 文件 |

---

## 16. 建议的 7 天学习路线

### 第 1 天：跑通

- 运行 `audit_spectra.py`；
- 运行 `make_spectral_dataset.py`；
- 打开一张热图；
- 运行 2 epoch 训练。

### 第 2 天：理解数据

- 用 `read_scan_txt` 打印一条扫描；
- 手工算一个 `raw - dark`；
- 手工算一列 z-score；
- 对照热图看材料行。

### 第 3 天：理解 CNN 形状

- 把第 6 节的 shape 表抄一遍；
- 用 `model_summary.txt` 对照；
- 改 `out_channels`，观察参数变化。

### 第 4 天：理解训练

- 读 `train.py`；
- 读 `Trainer.fit`；
- 读 `_train_epoch`；
- 找到 `backward`、`step`、`zero_grad`。

### 第 5 天：理解指标

- 打开混淆矩阵；
- 计算一个 precision/recall 例子；
- 看 `test_per_class.csv`。

### 第 6 天：做实验

- 比较 `smooth_window=1` 和 `9`；
- 比较 `blank_mode=mean` 和 `paired`；
- 比较 `strategy=blocked` 和 `stratified`。

### 第 7 天：做更严格的评估

- 按采集日期/批次重新划分；
- 记录每个实验的 `resolved_config.yaml`；
- 不要只看 accuracy，也看 macro-F1 和混淆矩阵。

---

## 17. 读完这张清单，你就算读懂了

你应该能回答：

1. 原始 txt 里有多少条扫描、多少个波长点？
2. 为什么 xlsx 少列时要用 txt？
3. 暗电流是按什么维度相减？
4. z-score 的均值和标准差沿哪个轴计算？
5. `none` 怎样参与相减？
6. 一个气味的 50 条扫描怎样变成 50 个样本？
7. `X.shape` 为什么是 `[2550,1,23,592]`？
8. 23 种材料是怎样取交集的？
9. CNN 每一层把 shape 变成什么？
10. 51 个 logits 怎样变成概率和类别？
11. `loss.backward()` 和 `optimizer.step()` 分别在做什么？
12. 为什么当前 100% 不等于跨批次 100%？
13. `best.pth` 里除了权重还有什么？
14. 怎样生成全部热图、重新训练和预测一个新气味？

如果这些问题都能用自己的话回答，你已经能独立阅读、修改和扩展这个项目了。


