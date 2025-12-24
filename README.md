# Kaggle ECG 数据生成与训练指南

## TPU 环境配置与训练指南 (Google Cloud TPU VM)

在新的 TPU VM 机器上，请按照以下步骤快速恢复环境并开始训练。

### 1. 环境安装 (TPU v6e 特别说明)
由于 TPU v6e 较新，建议使用以下命令安装最新支持的驱动和运行时：

```bash
# 1. 安装 PyTorch 2.5 和适配 v6e 的 libtpu
pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip install torchvision==0.20.0
pip install -r requirements.txt

# 2. 修复 PATH (如果你还没做过)
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
source ~/.bashrc
```

### 1.1 TPU 状态监控
由于驱动版本冲突，如果需要使用 `tpu-info` 查看芯片状态，必须安装 **0.3.0** 旧版本，并指定驱动路径：

```bash
# 1. 安装兼容版本
pip install --no-deps tpu-info==0.3.0

# 2. 从另一个终端窗口查看状态
export TPU_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
tpu-info
# 或者实时监控
watch -n 1 tpu-info
```

### 2. 数据准备

使用 `scripts/prepare_all_data.sh` 脚本可以自动完成以下工作：
1. 从 GCS 下载原始数据 (`ori_data`, `ori_csv`)。
2. 生成标准网格数据集 (`dataset_grid`).
3. 生成增强数据集 (`dataset_grid_moire_blur_required`).
4. 生成纯波形数据集 (`dataset_wave_only`).

```bash
# 一键准备所有数据
bash scripts/prepare_all_data.sh
```

*(如果只需下载原始数据，可手动运行 `gsutil -m cp -r gs://tpu-research-01-ecg-data/kaggle_ecg/ori_data .` 和 `gsutil -m cp -r gs://tpu-research-01-ecg-data/kaggle_ecg/ori_csv .`)*

### 3. 开始训练

使用适配 TPU 的训练脚本 `experiment/train_tpu.py`。
**注意：** 必须设置 `TPU_LIBRARY_PATH` 和 `PJRT_DEVICE` 环境变量，否则可能会报错 `libtpu not found` 或回退到 CPU。

```bash
# 核心环境变量配置
export TPU_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/libtpu/libtpu.so
export PJRT_DEVICE=TPU

# 推荐：使用 FP32 训练 (更稳定，防止 Loss NaN)
export XLA_USE_BF16=0 

# 如果追求极速，可尝试开启 BF16（需确保学习率够小）
# export XLA_USE_BF16=1

python experiment/train_tpu.py --config experiment/configs/grid_aug_net.yaml
```

#### 常见报错与解决
*   **Running on CPU (TfrtCpuClient created)**: 说明没有设置 `PJRT_DEVICE=TPU` 或驱动未找到。请检查 `TPU_LIBRARY_PATH` 是否指向了正确的 `libtpu.so`。
*   **Loss NaN**: 通常是因为学习率过大或者开启了 BF16 导致的数值溢出。建议设置 `XLA_USE_BF16=0` 并降低学习率。
*   **libpng error: Read Error**: 训练数据损坏。建议删除 `dataset_grid` 目录并使用 `scripts/build_dataset.py` 重新生成。

---

# Kaggle ECG 数据生成辅助说明

## 批量生成波形整页与掩模

使用 `scripts/batch_render_waveforms.py` 可以批量渲染 ECG 网格页以及对应的波形掩模。默认参数已经配置为：

- 行顶位置为 `(18, 25, 32, 39)` 个大格，对应前三行下移 3 格、第四行下移 2 格；
- 波形与校准方波共享基线，围绕各自红线±0.25 个大格抖动；
- 掩模使用与整页完全相同的随机种子，保证对齐。

运行示例（覆盖旧输出，可先清空目标目录）：

```bash
python scripts/batch_render_waveforms.py \
    --csv-root ecg_generator_with_train_src/train \
    --grid ecg_generator_with_train_src/export/clean_ecg_grid.png \
    --image-dir gen_wave/images \
    --mask-dir gen_wave/masks \
    --overwrite
```

这样会分别写入 `gen_wave/images/<id>.png`（整页）以及 `gen_wave/masks/<id>_mask.png`（波形掩模）。若不指定 `--image-dir` / `--mask-dir`，脚本仍会把两者放在 `--output-dir` 下。当 CSV 内存在 NaN 时，脚本会输出 `invalid value encountered in cast` 的提示，可忽略或后续清洗数据。

## 在主流程中使用波形数据

`scripts/build_dataset.py` 现已支持把 `gen_dataset/wave` 中的整页图作为额外输入，并在裁剪阶段使用对应的波形掩模来保证每个裁剪块都包含可见波形。关键参数：

- `--wave-root`: 指向包含 `<id>.png` 和 `<id>_mask.png` 成对文件的目录（会递归搜索 PNG）；
- `--wave-min-coverage`: 波形掩码覆盖率下限（默认 0.05，基于膨胀后的掩码像素占比）；
- `--wave-coverage-kernel`: 波形掩码在 coverage 计算前的膨胀核大小，默认 45，可根据波形线粗细调整。

使用示例（在原有 `ori_data` 的基础上额外混入 wave 数据，一并生成正负样本）：

```bash
python scripts/build_dataset.py \
    --input-root ori_data \
    --mask image_data/complete_mask.png \
    --wave-root gen_dataset/wave \
    --wave-min-coverage 0.05 \
    --wave-coverage-kernel 45 \
    --samples-per-image 10 \
    --negative-per-image 2 \
    --output-root dataset_grid_wave_mix \
    --overwrite \
    --limit 5
```

脚本会自动：

1. 为 `wave-root` 中的每张图片寻找同名 `_mask.png`（波形掩模），并在裁剪时以该掩模做 coverage 判定；
2. 仍然使用 `--mask` 指定的网格掩模作为监督标签，确保输出 mask 只包含网格；
3. 对 wave 样本禁用横向密度约束，并按照指定阈值/膨胀核强制裁剪到包含波形的区域。
