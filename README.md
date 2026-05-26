# FCN on PASCAL VOC 2012: Five-Class Semantic Segmentation

本项目用于完成语义分割实验：使用 FCN-8s 风格网络在 PASCAL VOC 2012 的五类子集上训练、评估并可视化结果，对比 `FCN+ResNet18` 与 `FCN+ResNet34`。扩展实验支持 `ResNet50` 编码器、可学习转置卷积上采样，以及包含背景的 VOC 21 类语义分割任务。

## 实验口径

输出类别为背景、人、车、猫、狗共 5 类。VOC 原始标签映射如下：

| VOC 标签 | 类别 | 训练标签 |
| ---: | --- | ---: |
| 0 | background | 0 |
| 15 | person | 1 |
| 7 | car | 2 |
| 8 | cat | 3 |
| 12 | dog | 4 |
| 其他类别与边界标签 | ignored | 255 |

默认配置 `filter_foreground: true`：在官方 `train` 和 `val` 划分内部，仅保留至少含有一个目标前景类别的图像；其他物体类别像素仍设为 `ignore_index=255`，不参与损失与指标计算。

模型使用 ImageNet 预训练的 ResNet 编码器。分割预测头由本项目初始化并在五类任务上训练，因此这里不是加载一个已经针对 VOC 训练过的完整 FCN 权重。

## 目录结构

```text
.
├── configs/                 # 两组公平对比配置
├── datasets/                # VOC 五类子集及同步数据增强
├── models/                  # FCN-ResNet18/34 实现
├── utils/                   # 指标、训练循环、速度测试、可视化等
├── prepare_data.py          # 下载/检查 VOC 数据
├── collect_environment.py   # 输出服务器软件及 GPU 环境
├── train.py                 # 训练入口
├── evaluate.py              # 评估及推理速度测试入口
├── predict.py               # 结果图生成入口
├── compare_models.py        # 对比表汇总入口
├── scripts/                 # 服务器运行脚本
└── report/实验报告模板.md    # 运行后填入真实数据
```

主要扩展配置如下：

```text
configs/fcn_resnet50.yaml          # 第三编码器对比，五类 + 双线性上采样
configs/fcn_resnet34_deconv.yaml   # 上采样对比，五类 + 转置卷积
configs/fcn_resnet34_voc21.yaml    # 完整 VOC 任务，背景 + 20 个前景类
scripts/run_extensions.sh          # 只训练三种新增实验的执行脚本
```

## 服务器环境安装

推荐 Python 3.10 或以上版本。服务器 CUDA 版本不同，建议先依据 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装适配 GPU 的 `torch` 和 `torchvision`，再安装其余依赖：

```bash
pip install -r requirements.txt
```

代码依赖 torchvision 官方提供的 [`VOCSegmentation`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VOCSegmentation.html) 数据集接口，以及 [`ResNet18_Weights`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) / `ResNet34_Weights` 编码器预训练权重接口。

## 数据准备

自动下载并显示筛选后的训练/验证样本数量：

```bash
bash scripts/download_voc.sh
```

也可以将数据手工放置在以下结构中：

```text
data/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/Segmentation/
            ├── train.txt
            └── val.txt
```

然后检查数据可读取：

```bash
python prepare_data.py --root ./data
```

## 训练

记录报告所需的服务器环境信息：

```bash
python collect_environment.py
```

信息将写入 `outputs/metrics/environment.json`。

两个实验除编码器外采用相同参数：`320x320` 输入、`batch_size=8`、`epochs=50`、`AdamW`、初始学习率 `1e-4`、余弦退火、相同增强和随机种子。相同形状的 FCN 预测头使用独立固定种子初始化；数据加载器也使用固定生成器种子，使两个编码器实验的训练乱序和增强随机序列可复现。

```bash
python train.py --config configs/fcn_resnet18.yaml
python train.py --config configs/fcn_resnet34.yaml
```

训练输出：

```text
outputs/checkpoints/fcn_resnet18_best.pth
outputs/checkpoints/fcn_resnet34_best.pth
outputs/logs/*_history.csv
outputs/metrics/*_train_summary.json
outputs/visualizations/*_curves.png
```

显存不足时可同时修改两个 YAML 中的 `batch_size`；公平对比要求两组配置保持一致。

## 评估与结果表

```bash
python evaluate.py --config configs/fcn_resnet18.yaml
python evaluate.py --config configs/fcn_resnet34.yaml
python compare_models.py
```

评估计算 Pixel Accuracy、mIoU、各类别 IoU，以参数及缓冲区所占存储量作为模型大小，并在同一验证子集上统计经过预热后的平均单图推理时间。汇总表生成在：

```text
outputs/metrics/model_comparison.md
outputs/metrics/model_comparison.csv
outputs/metrics/class_iou_comparison.csv
```

五类可视化中灰色区域为不参与损失和指标的 VOC 非目标类别或边界区域。模型在灰色区域上的颜色预测不能直接作为五类任务的错误证据。

## 推理可视化

生成三张具备目标类别覆盖性的比较图，每张图包含原图、真实标签和两个模型预测：

```bash
python predict.py --compare
```

固定选择验证样本时可指定索引：

```bash
python predict.py --compare --indices 0 10 20
```

输出目录为 `outputs/visualizations/comparison/`。

## 一次执行完整实验

数据准备完成后，可以在同一服务器/GPU 环境执行：

```bash
bash scripts/run_experiments.sh
```

如需指定设备：

```bash
DEVICE=cuda:0 bash scripts/run_experiments.sh
```

## 扩展实验

扩展实验仅新增三次训练：

| 实验 | 配置 | 对比目的 |
| --- | --- | --- |
| FCN-ResNet50-Bilinear-5Class | `configs/fcn_resnet50.yaml` | 与 ResNet18、ResNet34 对比精度、速度与大小 |
| FCN-ResNet34-Deconv-5Class | `configs/fcn_resnet34_deconv.yaml` | 与 ResNet34 双线性上采样对比 |
| FCN-ResNet34-Bilinear-VOC21 | `configs/fcn_resnet34_voc21.yaml` | 完成背景加 VOC 20 个前景类任务 |

`VOC21` 配置输出 `21` 类；仅 VOC 标签 `255` 作为忽略区域，不再忽略其他目标类别。

服务器需要保留基础实验的以下 checkpoint，以便扩展脚本复用基线进行对比：

```text
outputs/checkpoints/fcn_resnet18_best.pth
outputs/checkpoints/fcn_resnet34_best.pth
```

拉取扩展代码后运行：

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
DEVICE=cuda:0 bash scripts/run_extensions.sh
```

扩展结果输出：

```text
outputs/metrics/backbone_comparison.md
outputs/metrics/upsampling_comparison.md
outputs/metrics/voc21_results.md
outputs/visualizations/backbone_comparison/
outputs/visualizations/upsampling_comparison/
outputs/visualizations/fcn_resnet34_voc21/
```

若基础 checkpoint 已被清除，需先重新执行基础实验脚本，再执行扩展脚本。

扩展实验完成后，如需将表格与图片推回 GitHub 而不提交权重和数据集，在服务器项目目录执行：

```bash
git add -f outputs/metrics/backbone_comparison.md outputs/metrics/backbone_comparison.csv
git add -f outputs/metrics/backbone_class_iou_comparison.csv
git add -f outputs/metrics/upsampling_comparison.md outputs/metrics/upsampling_comparison.csv
git add -f outputs/metrics/upsampling_class_iou_comparison.csv
git add -f outputs/metrics/voc21_results.md outputs/metrics/voc21_results.csv
git add -f outputs/metrics/voc21_class_iou.csv
git add -f outputs/visualizations/fcn_resnet50_curves.png
git add -f outputs/visualizations/fcn_resnet34_deconv_curves.png
git add -f outputs/visualizations/fcn_resnet34_voc21_curves.png
git add -f outputs/visualizations/backbone_comparison/*.png
git add -f outputs/visualizations/upsampling_comparison/*.png
git add -f outputs/visualizations/fcn_resnet34_voc21/*.png
git commit -m "Add extension experiment results"
git push origin main
```

## 报告填写

在服务器完成运行后，将以下真实输出填入 [实验报告模板](report/实验报告模板.md)：

- Python、PyTorch、CUDA 和 GPU 信息；
- `outputs/metrics/model_comparison.md` 中的对比结果；
- `outputs/visualizations/comparison/` 中的 2 至 3 张结果图；
- 训练曲线与实际训练耗时；
- 基于可视化结果和各类 IoU 的分析。
- `outputs/metrics/backbone_comparison.md`、`upsampling_comparison.md` 与 `voc21_results.md` 中的扩展结果。
