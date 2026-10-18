# 配置驱动的通用卷积神经网络项目

这是一个面向图像分类的通用 CNN 工程。数据、网络结构、训练策略、反向传播参数、评估和预测都可以通过 YAML 配置控制；Python 代码中不散落数据集名称、层数、学习率等实验参数。

项目已经在 `dl1` 环境（Python 3.14.6、PyTorch 2.14.1 CUDA 13.0、torchvision 0.29.1）中跑通合成数据训练、断点保存和曲线绘制。

当前保留了一份最终版代码的真实运行结果：`outputs/runs/mnist_20261018_201403_verified_final/`。在 RTX 3060 Laptop GPU 上训练 MNIST 1 个 epoch，验证准确率 95.72%，测试准确率 96.24%，测试 macro-F1 96.22%；最佳权重、训练日志、曲线和混淆矩阵都在该目录中。1 个 epoch 只是快速验证，正式实验可使用 `configs/mnist_lenet.yaml` 默认的 3 个 epoch。

## 快速开始

```powershell
conda activate dl1
cd G:\Users\3229629598\desktop\dl\cnn_framework

# 1. 不下载数据，快速检查数据、网络、前向和反向传播
python train.py --config configs/smoke_test.yaml --dry-run

# 2. 合成数据完整训练（2 个 epoch，适合确认整套流程）
python train.py --config configs/smoke_test.yaml

# 3. 真实 MNIST 训练
python train.py --config configs/mnist_lenet.yaml

# 4. 临时覆盖配置，不修改 YAML
python train.py --config configs/cifar10_resnet.yaml --set train.epochs=2 dataset.num_workers=0
```

如果 PowerShell 中 `conda activate` 不可用，可以使用：

```powershell
conda run -n dl1 python train.py --config configs/smoke_test.yaml
```

在某些 Windows 终端中，`conda run` 可能因中文控制台编码报 `UnicodeEncodeError`；直接使用 `dl1` 环境里的 Python 解释器即可：

```powershell
& 'C:\Users\3229629598\miniconda3\envs\dl1\python.exe' train.py --config configs/smoke_test.yaml
```

## 项目结构

```text
cnn_framework/
├─ configs/
│  ├─ default.yaml              # 完整默认配置，所有选项和默认值都在这里
│  ├─ smoke_test.yaml           # 合成数据冒烟配置
│  ├─ mnist_lenet.yaml          # MNIST + LeNet 风格网络
│  ├─ cifar10_resnet.yaml       # CIFAR-10 + 可配置残差网络
│  └─ custom_image_folder.yaml  # 自定义 ImageFolder 数据集
├─ cnnlab/
│  ├─ config.py                 # 配置合并、覆盖、校验、快照
│  ├─ datasets.py               # 数据集、数据增强、DataLoader
│  ├─ layers.py                 # 可配置层、激活、归一化、残差块
│  ├─ model.py                  # 动态 CNN 组装与结构摘要
│  ├─ losses.py                 # 分类损失与类别权重
│  ├─ metrics.py                # 准确率、top-k、逐类指标、混淆矩阵
│  ├─ engine.py                 # 训练/验证、AMP、反向传播、断点、早停
│  └─ utils.py                  # 种子、设备、日志、路径等工具
├─ train.py                     # 训练入口
├─ evaluate.py                  # 评估入口
├─ predict.py                   # 单图/目录预测入口
├─ tests/test_smoke.py          # 单元冒烟测试
├─ requirements.txt
└─ outputs/                     # 训练日志、权重、曲线、评估结果
```

## 配置体系

加载顺序是：

```text
configs/default.yaml  <-  configs/场景配置.yaml  <-  命令行 --set
```

注意：

- 字典会递归合并，列表会整体替换。例如场景配置里的 `model.layers` 会替换默认网络。
- 配置中的相对路径相对于项目根目录 `cnn_framework/`，不是命令行当前目录。
- 每次运行会生成 `outputs/runs/数据集名_时间_运行名/`，其中保存最终生效的 `resolved_config.yaml`、日志、权重、曲线和指标。
- 所有可调参数都在 `configs/default.yaml` 中列出了默认值和中文注释。新增配置项时，建议先加到默认配置，再在代码中读取。

### 命令行覆盖

```powershell
python train.py --config configs/cifar10_resnet.yaml --set train.epochs=5 dataset.batch_size=64
```

值按 YAML 语法解析，因此布尔值写 `true/false`，空值写 `null`。列表/嵌套层结构通常直接编辑场景 YAML 更清晰。

## 数据集配置

`dataset.name` 支持：

| 名称 | 说明 |
|---|---|
| `mnist` | 手写数字，1 通道，10 类 |
| `fashion_mnist` | Fashion-MNIST，1 通道，10 类 |
| `kmnist` | Kuzushiji-MNIST，1 通道，10 类 |
| `cifar10` | CIFAR-10，3 通道，10 类 |
| `cifar100` | CIFAR-100，3 通道，100 类 |
| `image_folder` | 自定义目录，每个子目录为一个类别 |
| `synthetic` | 内置彩色/灰度图案合成数据，不需要下载 |

`synthetic` 直接生成 `[C,H,W]` 浮点张量，因此不会应用 `dataset.transforms`；它只用于检查代码和训练循环，不代表真实视觉任务难度。真实任务请使用内置数据集或 `image_folder`。

常用数据参数：

| 参数 | 作用 |
|---|---|
| `root` | 数据根目录 |
| `download` | 是否允许 torchvision 下载 |
| `num_classes` | 类别数；`image_folder` 可写 `null`/`auto` 自动推断 |
| `in_channels` | 输入通道数，必须与网络输入一致 |
| `image_size` | 变换后的 `[H,W]` 或正方形边长 |
| `batch_size` | batch 大小 |
| `num_workers` | DataLoader 进程数，Windows 先用 0 最稳 |
| `pin_memory` | 锁页内存，GPU 训练建议开启 |
| `persistent_workers` | 是否复用 DataLoader worker |
| `prefetch_factor` | 每个 worker 预取的 batch 数 |
| `drop_last` | 是否丢弃最后不足一个 batch 的训练数据 |
| `val_split` | 从训练集划分验证集的比例 |
| `use_test_as_val` | 是否拿测试集当验证集；正式实验不建议开启 |
| `mean` / `std` | 数据集级均值/方差，配合 `normalize.mode=custom` |

### 数据增强

`dataset.transforms.train` 和 `dataset.transforms.val` 分别控制训练和评估。训练常用随机增强；验证/测试必须保持确定性。

| 选项 | 说明 |
|---|---|
| `auto_augment` | `null`/`imagenet`/`cifar10`/`svhn` |
| `resize` | 统一缩放，整数或 `[H,W]` |
| `random_resized_crop` | 随机缩放裁剪，含 `enabled/size/scale/ratio` |
| `random_crop` | 随机裁剪，含 `size/padding/padding_mode/fill` |
| `center_crop` | 固定中心裁剪 |
| `random_horizontal_flip` | 水平翻转概率 |
| `random_vertical_flip` | 垂直翻转概率 |
| `random_rotation` | 随机旋转角度，或带插值参数的字典 |
| `random_affine` | 平移、缩放、剪切、旋转 |
| `color_jitter` | 亮度、对比度、饱和度、色相 |
| `grayscale` | 随机灰度化，输出通道数必须匹配网络 |
| `gaussian_blur` | 高斯模糊 |
| `to_tensor` | 必须为 `true` |
| `normalize.mode` | `auto`/`custom`/`none` |
| `random_erasing` | 张量随机擦除 |

## 网络结构配置

网络由 `model.layers`（特征提取）和 `model.head`（分类头）两个列表组成。构建器会逐层推导形状，`linear` 的输入特征数、卷积的输入通道数不需要手工填写。

每个层块由以下部分组成：

```text
core -> norm -> activation -> dropout
```

顺序可由 `order` 修改，例如：

```yaml
order: [core, activation, norm, dropout]
```

### 支持的核心类型

| `type` | 主要参数 |
|---|---|
| `conv2d` | `out_channels/kernel_size/stride/padding/dilation/groups/bias/padding_mode/depthwise/depth_multiplier` |
| `linear` | `out_features`（可写 `num_classes`）、`bias` |
| `maxpool2d` | `kernel_size/stride/padding/ceil_mode` |
| `avgpool2d` | 同上，另有 `count_include_pad/divisor_override` |
| `adaptive_avgpool2d` | `output_size` |
| `adaptive_maxpool2d` | `output_size` |
| `global_avgpool` | 输出 `[C]` |
| `global_maxpool` | 输出 `[C]` |
| `flatten` | `start_dim` |
| `identity` | 无 |
| `sequence` | `layers` 子列表 |
| `residual` | `layers`、`projection`、`projection_norm`、`projection_bias` |
| `activation` | `name` 及该激活函数的参数 |
| `dropout`/`dropout2d`/`alpha_dropout` | `p` |
| `batchnorm2d`/`batchnorm1d` | 该归一化层的参数 |
| `instancenorm2d`/`layernorm`/`groupnorm` | 对应归一化层参数 |

`conv2d` 的 `padding` 支持整数、`[H,W]`、`same`、`valid`；`groups` 支持整数、`depthwise`、`auto`。`same` 在 stride>1 时会按输入尺寸动态做非对称填充。

### 激活函数

`activation` 支持字符串或字典：

```yaml
activation: relu
activation: {name: leaky_relu, negative_slope: 0.1}
activation: {name: gelu, approximate: tanh}
activation: null          # 显式关闭
```

可用名称：`relu`、`relu6`、`leaky_relu`、`elu`、`selu`、`celu`、`gelu`、`silu`/`swish`、`mish`、`tanh`、`sigmoid`、`hardsigmoid`、`hardswish`、`softplus`、`softsign`、`prelu`、`logsigmoid`、`softmax`、`log_softmax`、`none`/`identity`。使用 `nll` 损失时，网络末端应配置 `log_softmax`。

### 归一化

`norm` 支持字符串或字典：

```yaml
norm: batchnorm2d
norm: {type: batchnorm2d, eps: 1.0e-5, momentum: 0.1}
norm: {type: groupnorm, num_groups: 8}
norm: {type: layernorm}
norm: null
```

可用类型：`batchnorm2d`、`batchnorm1d`、`instancenorm2d`、`layernorm`、`groupnorm`、`none`。

### 残差块

```yaml
- type: residual
  repeat: 2
  projection: auto          # auto/identity/conv1x1/linear/none
  projection_norm: {type: batchnorm2d}
  activation: {name: relu, inplace: true}   # 相加后的激活
  layers:
    - type: conv2d
      out_channels: 64
      kernel_size: 3
      padding: same
      norm: {type: batchnorm2d}
      activation: null
    - type: conv2d
      out_channels: 64
      kernel_size: 3
      padding: same
      norm: {type: batchnorm2d}
      activation: null
```

内部最后一个卷积通常不激活，等主分支和捷径相加后再统一激活。`projection: auto` 会在形状不一致时自动创建 1x1 卷积（4 维特征）或线性投影（2 维特征）。

### 层块通用选项

| 选项 | 说明 |
|---|---|
| `repeat` | 将同一层配置连续构建多次 |
| `trainable` | `false` 冻结该层所有参数 |
| `checkpoint` | 开启梯度检查点，省显存、增加计算 |
| `weight_init` | `default/kaiming_normal/kaiming_uniform/xavier_normal/xavier_uniform/normal/uniform/orthogonal/dirac/zeros/ones` |
| `order` | `core/norm/activation/dropout` 的执行顺序 |
| `norm` | 归一化配置或 `null` |
| `activation` | 激活配置或 `null` |
| `dropout` | Dropout 概率或字典 |

`model.defaults` 可以设置所有层的默认激活、归一化、Dropout、执行顺序、梯度检查点和初始化。层内写同名键会覆盖默认值；显式写 `null` 表示关闭。

## 前向与反向传播参数

前向传播由 `model.layers`、`model.head`、`activation`、`norm`、`dropout`、`residual`、`channels_last` 控制。残差相加、池化、展平等都是标准 PyTorch 运算，自动求导会构建对应计算图。

反向传播相关配置集中在 `train`、`loss`、`optimizer` 和 `scheduler`：

| 配置 | 作用 |
|---|---|
| `train.grad_accumulation_steps` | 梯度累积步数，模拟更大 batch |
| `train.backward.grad_clip.enabled` | 是否裁剪梯度 |
| `train.backward.grad_clip.mode` | `norm`（按范数）/`value`（按数值）/`none` |
| `train.backward.grad_clip.max_norm` | 范数裁剪阈值 |
| `train.backward.grad_clip.value` | 数值裁剪阈值 |
| `train.backward.grad_clip.norm_type` | 梯度范数类型，常用 2 |
| `train.amp.enabled` | 是否启用混合精度（仅 CUDA 生效） |
| `train.amp.dtype` | `float16` 或 `bfloat16` |
| `train.amp.grad_scaler` | fp16 的缩放器参数 |
| `train.stop_on_nan` | 损失为 NaN/Inf 时停止还是跳过 batch |
| `train.log_grad_norm` | 是否记录梯度范数 |
| `layer.checkpoint` | 梯度检查点，反向时重算中间激活 |

### 损失函数

`loss.name` 支持 `cross_entropy`、`bce_with_logits`、`nll`。`loss.params` 可配置 `weight`、`ignore_index`、`reduction`、`label_smoothing`、`pos_weight`。`weight: balanced` 会按训练集类别频率自动计算权重。

### 优化器

`optimizer.name` 支持 `sgd`、`adam`、`adamw`、`rmsprop`、`adagrad`、`adadelta`。`optimizer.params` 原样传给对应的 `torch.optim` 构造函数，例如：

```yaml
optimizer:
  name: sgd
  lr: 0.05
  weight_decay: 0.0005
  params:
    momentum: 0.9
    nesterov: true
```

`no_weight_decay_for_bias_and_norm: true` 会自动把 bias 和归一化层放进无权重衰减组。也可以使用 `optimizer.param_groups` 按正则表达式自定义分组：

```yaml
param_groups:
  - name: no_decay
    patterns: ["\\.bias$", "norm"]
    lr_multiplier: 1.0
    weight_decay: 0.0
```

### 学习率调度

`scheduler.name` 支持 `none`、`step`、`multi_step`、`cosine`、`cosine_restarts`、`plateau`、`linear`、`exponential`、`one_cycle`。`step_on` 可选 `epoch` 或 `batch`；`plateau` 只支持按 epoch，并按 `train.monitor` 的指标更新；`one_cycle` 必须配合 `step_on: batch`。

### 早停与断点

`train.early_stopping` 控制耐心值和最佳权重恢复；`checkpoint` 控制 best/last/周期保存和 `resume_from` 续训。断点中包含模型、优化器、调度器、GradScaler、历史记录、类别映射和配置。

### 运行、日志、断点与预测参数

| 配置段 | 常用选项 |
|---|---|
| `runtime` | `output_dir/device/seed/deterministic/benchmark/float32_matmul_precision/dry_run` |
| `runtime.compile` | `enabled/backend/mode/fullgraph/dynamic` |
| `checkpoint` | `save_best/best_name/save_last/last_name/save_every_n_epochs/resume_from` |
| `logging` | `run_name/unique_run_dir/log_file/console_level/file_level/plot_curves` |
| `evaluation` | `topk/binary_threshold` |
| `predict` | `checkpoint/source/output_csv/topk/batch_size` |

`dataset.synthetic` 的专用参数是 `channels/num_classes/image_size/train_samples/val_samples/test_samples/noise_std`。

## 评估与预测

```powershell
# 评估验证集/测试集
python evaluate.py --config configs/mnist_lenet.yaml `
  --checkpoint outputs/runs/mnist_xxx/best.pth --split test

# 单张图片
python predict.py --config configs/custom_image_folder.yaml `
  --checkpoint outputs/runs/custom_xxx/best.pth --source data/test.jpg

# 整个目录
python predict.py --checkpoint outputs/runs/custom_xxx/best.pth --source data/test_images
```

不提供 `--config` 时，预测脚本会使用断点中保存的配置。`predict.py` 会把预处理输出形状与断点输入形状核对，避免训练/推理预处理不一致。

## 输出文件

每次训练目录包含：

- `resolved_config.yaml` / `.json`：最终生效配置
- `environment.json`：Python、PyTorch、CUDA 版本
- `model_summary.txt`：每层形状和参数量
- `dataset_meta.json`：类别映射、均值方差、损失配置
- `train.log`：完整中文日志
- `history.csv` / `history.json`：逐 epoch 指标
- `curves.png`：loss、accuracy、macro-F1、学习率曲线
- `best.pth` / `last.pth`：最佳和最后断点
- `test_metrics.json`、`test_per_class.csv`、`test_confusion.csv/png`：测试指标

## 添加新层

1. 在 `cnnlab/layers.py` 的 `_build_core` 中增加一个分支，返回 `(nn.Module, 输出形状, 子层摘要)`。
2. 在 `configs/default.yaml` 的层说明和 README 中补充参数。
3. 如果新层改变形状，务必同时返回正确的输出形状；`build_layers` 会用它推导下一层。
4. 如果是复合层，内部通过 `build_layers` 递归构建，就能继续使用激活、归一化、初始化和冻结等通用能力。

## 测试

```powershell
python -m unittest discover -s tests -v
```

`tests/test_smoke.py` 不下载数据，会检查：

- YAML 合并与配置校验；
- 合成数据集构建；
- 动态 CNN 前向；
- 反向传播；
- 残差网络形状推导；
- 一个 batch 的损失/梯度是否有限。

