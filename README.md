# Kaggle ECG Analysis & Segmentation

本项目用于生成合成 ECG 数据并训练分割模型（支持网格与波形的像素级分割），专为 TPU 环境优化。

## 1. 环境配置 (TPU)

在 TPU VM 上，必须安装支持 TPU 的 PyTorch 版本 (`torch_xla`)。

```bash
# 1. 卸载可能存在的非 TPU 版本 PyTorch
pip uninstall -y torch torchvision

# 2. 安装 TPU 版本的 PyTorch (v2.5.0)
pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html

# 3. 安装其他依赖
pip install -r requirements.txt numpy
```

## 2. 数据准备

### 2.1 下载原始数据
原始 CSV 数据存储在 GCS 上。

```bash
# 确保已认证 gcloud auth login
# 下载并解压数据到 /home/creart/kaggle_ecg/ori_csv
bash scripts/prepare_original_data.sh  # (需自行编写或手动 gsutil cp)
# 或者手动:
mkdir -p ori_csv
gsutil cp -r gs://tpu-research-01-ecg-data/kaggle_ecg/ori_csv/* ori_csv/
cd ori_csv && tar -zxvf train-csvs.tar.gz && cd ..
```

### 2.2 生成合成波形与掩码
使用 `batch_render_waveforms.py` 批量生成波形图像和对应的波形掩码。

```bash
python scripts/batch_render_waveforms.py \
    --csv-root ori_csv/train-csvs \
    --grid ecg_generator_with_train_src/export/clean_ecg_grid.png \
    --image-dir data/wave_images \
    --mask-dir data/wave_masks
```

### 2.3 构建训练数据集 (3分类)
将波形与背景网格混合，生成最终的训练样本。
*   **Target**: `dataset_grid_wave_mix`
*   **Classes**: 0 (背景), 1 (网格), 2 (波形)

```bash
python scripts/build_dataset.py \
    --input-root ori_csv/train \
    --mask image_data/complete_mask.png \
    --wave-root data/wave_images \
    --output-root dataset_grid_wave_mix \
    --overwrite \

```

### 2.4 下载样本数据 (Utilities)

如果需要下载少量真实数据用于测试或推理验证，可以使用 `scripts/download_subset.py`。
该脚本会从 Kaggle 竞赛 `physionet-ecg-image-digitization` 中随机下载指定数量的样本。

**使用示例：**
```bash
# 下载 10 个随机样本到 downloaded_subset/ 目录
python scripts/download_subset.py
```

**依赖条件：**
*   需要在 `~/.kaggle/kaggle.json` 配置好 Kaggle API Token。
*   需要预先接受该竞赛的 Rules。

## 3. 模型训练 (TPU)

使用 `experiment/train_tpu.py` 在 TPU 上进行多核并行训练。

### 3.1 关键特性
*   **8 卡并行**: 自动检测并利用 8 个 TPU 核心。
*   **自适应学习率**: 集成 `ReduceLROnPlateau`，当 Loss 不下降时自动降低 LR。
*   **鲁棒性**: 包含 NaN 检查和自动重试机制。

### 3.2 启动训练

**标准启动命令 (推荐)**:

```bash
# 设置 TPU 设备环境
export PJRT_DEVICE=TPU

# 启动训练
# 默认使用 8 核心，配置文件对应 3 分类任务
python experiment/train_tpu.py --config experiment/configs/grid_wave_net.yaml
```

**如遇 NaN 不稳定**:
如果在训练初期遇到 `train_loss: nan`，可以尝试关闭 BFloat16 加速（虽然这会降低速度）：

```bash
unset XLA_USE_BF16
python experiment/train_tpu.py --config experiment/configs/grid_wave_net.yaml
```

### 3.3 监控
*   **查看 TPU 占用**: `/home/creart/.local/bin/tpu-info`
*   **查看日志**: 训练过程会实时打印 Loss 和 LR 更新信息。日志和 Checkpoint 保存在 `runs/grid_wave_mix_run/`。

## 4. 故障排除

*   **Unsupported nprocs (4)**: 忽略此警告，脚本会自动适配为 8 卡 (v2/v3) 或单卡 (v6e-1)。
*   **Processes Stuck**: 如果训练意外中断，部分后台进程可能残留，导致下一次无法启动。请运行 `killall python` 清理环境。
