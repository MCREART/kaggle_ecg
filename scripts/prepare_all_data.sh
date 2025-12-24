#!/bin/bash
set -e

# 1. 下载数据 (如果尚未下载)
if [ ! -d "ori_data" ]; then
    echo "正在下载 ori_data..."
    gsutil -m cp -r gs://tpu-research-01-ecg-data/kaggle_ecg/ori_data .
fi

if [ ! -d "ori_csv" ]; then
    echo "正在下载 ori_csv..."
    gsutil -m cp -r gs://tpu-research-01-ecg-data/kaggle_ecg/ori_csv .
fi

# 2. 生成标准网格数据集
echo "正在生成 dataset_grid (标准)..."
python scripts/build_dataset.py \
    --input-root ori_data \
    --output-root dataset_grid \
    --module-set all \
    --samples-per-image 5 \
    --overwrite

# 3. 生成强制摩尔纹模糊数据集
echo "正在生成 dataset_grid_moire_blur_required..."
python scripts/build_dataset.py \
    --input-root ori_data \
    --output-root dataset_grid_moire_blur_required \
    --module-set moire-blur-required \
    --samples-per-image 5 \
    --overwrite

# 4. 生成波形数据 (Waveforms)
if [ ! -d "gen_wave" ]; then
    echo "正在渲染波形中间数据..."
    # 确保有网格背景图
    GRID_IMG="ecg_generator_with_train_src/export/clean_ecg_grid.png"
    if [ ! -f "$GRID_IMG" ]; then
        echo "错误: 未找到 $GRID_IMG"
        exit 1
    fi
    
    python scripts/batch_render_waveforms.py \
        --csv-root ori_csv \
        --grid "$GRID_IMG" \
        --output-dir gen_wave \
        --overwrite
fi

# 5. 生成纯波形数据集
# 创建一个空目录作为 input-root 以避免混入普通网格图
mkdir -p empty_placeholder
echo "正在生成 dataset_wave_only..."
python scripts/build_dataset.py \
    --input-root empty_placeholder \
    --wave-root gen_wave \
    --output-root dataset_wave_only \
    --module-set all \
    --samples-per-image 5 \
    --overwrite

echo "数据准备完成！"
