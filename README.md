# Kaggle ECG Analysis & Segmentation

This project is used to generate synthetic ECG data and train segmentation models (supporting pixel-level segmentation of grids and waveforms), optimized for TPU environments.

## 1. Environment Configuration (TPU)

On TPU VMs, a PyTorch version that supports TPU (`torch_xla`) must be installed.

```bash
# 1. Uninstall any existing non-TPU version of PyTorch
pip uninstall -y torch torchvision

# 2. Install TPU version of PyTorch (v2.5.0)
pip install torch~=2.5.0 torch_xla[tpu]~=2.5.0 torchvision -f https://storage.googleapis.com/libtpu-releases/index.html

# 3. Install other dependencies
pip install -r requirements.txt numpy
```

## 2. Data Preparation

### 2.1 Download Original Data
Original CSV data is stored on GCS.

```bash
# Ensure you are authenticated with gcloud auth login
# Download and extract data to /home/creart/kaggle_ecg/ori_csv
bash scripts/prepare_original_data.sh  # (You need to write this or use gsutil cp manually)
# Or manually:
mkdir -p ori_csv
gsutil cp -r gs://tpu-research-01-ecg-data/kaggle_ecg/ori_csv/* ori_csv/
cd ori_csv && tar -zxvf train-csvs.tar.gz && cd ..
```

### 2.2 Generate Synthetic Waveforms and Masks
Use `batch_render_waveforms.py` to batch generate waveform images and corresponding waveform masks.

```bash
python scripts/batch_render_waveforms.py \
    --csv-root ori_csv/train \
    --grid ecg_generator_with_train_src/export/clean_ecg_grid.png \
    --image-dir data/wave_images \
    --mask-dir data/wave_masks
```

### 2.3 Build Training Dataset (5 Classes)
Mix waveforms with background grids to generate the final training samples.
*   **Target**: `dataset_grid_wave_mix`
*   **Classes**: 0 (BG), 1 (H-Line), 2 (V-Line), 3 (Inter), 4 (Wave)

```bash
python scripts/build_dataset.py \
    --input-root ori_csv/train \
    --mask image_data/complete_mask.png \
    --wave-root data/wave_images \
    --wave-mask-root data/wave_masks \
    --output-root dataset_grid_wave_mix \
    --overwrite
```

### 2.4 Download Sample Data (Utilities)

If you need to download a small amount of real data for testing or inference verification, you can use `scripts/download_subset.py`.
This script will randomly download a specified number of samples from the Kaggle competition `physionet-ecg-image-digitization`.

**Usage Example:**
```bash
# Download 10 random samples to the downloaded_subset/ directory
python scripts/download_subset.py
```

**Prerequisites:**
*   You need to configure the Kaggle API Token in `~/.kaggle/kaggle.json`.
*   You need to accept the Rules of the competition in advance.

## 3. Model Training (TPU)

Use `experiment/train_tpu.py` for multi-core parallel training on TPU.

### 3.1 Key Features
*   **8-Card Parallel**: Automatically detects and utilizes 8 TPU cores.
*   **Adaptive Learning Rate**: Integrates `ReduceLROnPlateau`, automatically lowering LR when Loss stops decreasing.
*   **Robustness**: Includes NaN checks and automatic retry mechanisms.

### 3.2 Start Training

**Standard Start Command (Recommended)**:

```bash
# Set TPU device environment
export PJRT_DEVICE=TPU

# Start training
# Default uses 8 cores, config file corresponds to 3-class task
python experiment/train_tpu.py --config experiment/configs/grid_wave_net.yaml
```

**If NaN Instability Occurs**:
If `train_loss: nan` is encountered early in training, try disabling BFloat16 acceleration (although this will reduce speed):

```bash
unset XLA_USE_BF16
python experiment/train_tpu.py --config experiment/configs/grid_wave_net.yaml
```

### 3.3 Monitoring
*   **View TPU Usage**: `/home/creart/.local/bin/tpu-info`
*   **View Logs**: Loss and LR update information are printed in real-time during training. Logs and Checkpoints are saved in `runs/grid_wave_mix_run/`.

## 4. Troubleshooting

*   **Unsupported nprocs (4)**: Ignore this warning; the script will automatically adapt to 8 cards (v2/v3) or single card (v6e-1).
*   **Processes Stuck**: If training is interrupted unexpectedly, some background processes might remain, preventing the next start. Please run `killall python` to clean up the environment.

