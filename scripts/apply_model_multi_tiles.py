from __future__ import annotations
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对整幅高分辨率心电图照片采用多种 tile 尺寸滑窗推理，并将结果拼回原图。
包含自动分辨率调整和自动方向矫正功能。
"""

"""
对整幅高分辨率心电图照片采用多种 tile 尺寸滑窗推理，并将结果拼回原图。

示例：
python scripts/apply_model_multi_tiles.py \
    --run-dir runs/grid_aug/unet_efficientnetb0 \
    --input-dir data_to_process \
    --output-dir model_outputs_multi_tiles \
    --tile-sizes 128 256 512 1024 \
    --overlap-frac 0.125
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from experiment.models_factory import ModelSpec, build_model, iter_model_specs  # type: ignore


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多 tile 尺寸滑窗推理")
    parser.add_argument("--run-dir", type=Path, required=True, help="训练输出目录（含 best.ckpt）")
    parser.add_argument("--checkpoint", type=Path, default=None, help="可选，自定义 checkpoint 路径")
    parser.add_argument("--input-dir", type=Path, required=True, help="待处理图像目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出结果目录")
    parser.add_argument(
        "--tile-sizes",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="滑窗 tile 边长列表",
    )
    parser.add_argument(
        "--overlap-frac",
        type=float,
        default=0.125,
        help="Tile 重叠比例（与 tile 尺寸相乘得到像素数）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="二值掩码阈值",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=2200,
        help="如果输入图像宽度超过此值，则自动缩放到此宽度进行推理，结果再放回原尺寸（0 表示不缩放）",
    )
    parser.add_argument(
        "--gray",
        action="store_true",
        help="是否在推理前将输入图像转为灰度图（然后再转回3通道以匹配模型输入）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="执行设备（默认自动选择 CUDA 或 CPU）",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_model_spec(key: str) -> ModelSpec:
    for spec in iter_model_specs():
        if spec.key == key:
            return spec
    raise KeyError(f"未知模型 key: {key}")


import yaml

def load_model(run_dir: Path, checkpoint: Path | None, device: torch.device) -> tuple[nn.Module, dict]:
    # Try finding config in run_dir or parent
    candidates = [
        run_dir / "config.json",
        run_dir / "config.yaml",
        run_dir / "config_tpu.yaml",
        run_dir.parent / "config.json",
        run_dir.parent / "config.yaml",
        run_dir.parent / "config_tpu.yaml",
    ]
    
    config_path = None
    for p in candidates:
        if p.exists():
            config_path = p
            break
    
    if config_path is None:
        raise FileNotFoundError(f"未找到配置文件 (config.json/yaml)，已搜索: {[str(p) for p in candidates]}")
    
    print(f"Loading config from: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        if config_path.suffix in {".yaml", ".yml"}:
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
            
    # Handle different config structures
    # train_tpu.py saves the full config structure: {models: [...], ...}
    # This script expects cfg["model"] ... let's adapt
    
    if "models" in cfg:
        # It's likely the multi-model config from train_tpu.py
        # We need to find the model spec that matches the current run directory name
        # run_dir usually ends with the model key (e.g. unet_efficientnetb0)
        current_model_key = run_dir.name
        model_cfg = None
        for m in cfg["models"]:
            if m["key"] == current_model_key:
                model_cfg = m
                break
        
        if model_cfg is None:
            # Fallback: just take the first one
            print(f"[WARN] Could not match model key '{current_model_key}' in config. Using first model.")
            model_cfg = cfg["models"][0]
            
        # Reconstruct the structure expected below
        # The script below uses model_cfg["spec"]["key"] but our config has model_cfg["key"] directly.
        # Let's map it.
        # And construct arguments.
        spec = resolve_model_spec(model_cfg["key"])
        
        # Check if 'classes' is in data config or model config
        # in train_tpu.py config: classes is in 'data' or 'models'
        num_classes = model_cfg.get("classes", cfg.get("data", {}).get("classes", 1))
        
        model = build_model(
            spec,
            in_channels=model_cfg.get("in_channels", 3),
            classes=num_classes,
            encoder_weights=None, # Inference doesn't need pretrained weights download usually, or it's handled
            **model_cfg.get("model_kwargs", {}),
        ).to(device)
        
        # Inject classes into cfg for later use in main
        if "model" not in cfg:
             cfg["model"] = {}
        cfg["model"]["classes"] = num_classes

    else:
        # Legacy config structure
        model_cfg = cfg["model"]
        spec = resolve_model_spec(model_cfg["spec"]["key"])
        model = build_model(
            spec,
            in_channels=model_cfg["in_channels"],
            classes=model_cfg["classes"],
            encoder_weights=None,
            **model_cfg.get("model_kwargs", {}),
        ).to(device)

    ckpt_path = checkpoint or (run_dir / "best.ckpt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")
    
    print(f"Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, cfg


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def tile_inference(
    model: nn.Module,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device,
    num_classes: int,
    use_gray: bool = False,
) -> np.ndarray:
    h, w = image.shape[:2]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap 需小于 tile_size")

    pad_h = int(np.ceil((h - overlap) / stride) * stride + overlap)
    pad_w = int(np.ceil((w - overlap) / stride) * stride + overlap)
    pad_bottom = max(0, pad_h - h)
    pad_right = max(0, pad_w - w)
    
    padded = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )

    # (H, W, C)
    prob_map = np.zeros((padded.shape[0], padded.shape[1], num_classes), dtype=np.float32)
    weight_map = np.zeros((padded.shape[0], padded.shape[1]), dtype=np.float32)

    for y in range(0, padded.shape[0] - tile_size + 1, stride):
        for x in range(0, padded.shape[1] - tile_size + 1, stride):
            tile = padded[y : y + tile_size, x : x + tile_size]
            
            if use_gray:
                tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                tile = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)
                
            tensor = preprocess_image(tile).to(device)
            with torch.no_grad():
                logits = model(tensor)
                # handle both binary and multi-class
                if num_classes == 1:
                    probs = torch.sigmoid(logits) # (B, 1, H, W)
                    pred = probs.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() # (H, W, 1)
                else:
                    probs = torch.softmax(logits, dim=1) # (B, C, H, W)
                    pred = probs.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() # (H, W, C)

            # Ensure broadcasting works if necessary, but permute already handles it
            if pred.ndim == 2:
                pred = pred[..., None]
                
            prob_map[y : y + tile_size, x : x + tile_size] += pred
            weight_map[y : y + tile_size, x : x + tile_size] += 1.0

    weight_map = weight_map[..., None] # (H, W, 1)
    weight_map[weight_map == 0] = 1.0
    prob_map /= weight_map
    return prob_map[:h, :w, :]


def overlay_classes(image: np.ndarray, pred_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    pred_map: (H, W) containing class indices (0, 1, 2)
    """
    overlay = image.copy()
    
    # Class 1: Grid (Red)
    mask_grid = (pred_map == 1)
    overlay[mask_grid] = (overlay[mask_grid] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
    
    # Class 2: Wave (Green)
    mask_wave = (pred_map == 2)
    overlay[mask_wave] = (overlay[mask_wave] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)
    
    return overlay


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.run_dir, args.checkpoint, device)
    
    classes = cfg["model"].get("classes", 1)
    print(f"检测模型类别数: {classes}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in args.input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"{args.input_dir} 中未找到 PNG/JPG 图像")

    for path in image_paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] 跳过无法读取的文件: {path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = image_rgb.shape[:2]
        
        # Check resizing
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # --- 1. Vertical -> Horizontal Check ---
        h, w = image_rgb.shape[:2]
        is_vertical = h > w
        if is_vertical:
            print(f"[INFO] 图像为竖向 ({w}x{h})，逆时针旋转90度为横向...")
            # Rotate 90 deg Counter-Clockwise (or Clockwise, doesn't matter much as long as it becomes horizontal)
            # Standard is usually Counter-Clockwise to verify. Let's stick to valid OpenCV consts.
            # ROTATE_90_COUNTERCLOCKWISE
            image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Update H, W after potential rotation
        original_h, original_w = image_rgb.shape[:2]
        
        # --- 2. Resize Logic ---
        # Check resizing (Always resize to standard width for inference stability)
        resize_scale = 1.0
        process_image = image_rgb
        if args.resize_width > 0 and original_w > args.resize_width:
            resize_scale = args.resize_width / float(original_w)
            new_h = int(original_h * resize_scale)
            # Use INTER_AREA for shrinking to avoid aliasing
            process_image = cv2.resize(image_rgb, (args.resize_width, new_h), interpolation=cv2.INTER_AREA)
            print(f"[INFO] Auto-resizing {path.name}: {original_w}x{original_h} -> {args.resize_width}x{new_h}")

        for tile_size in args.tile_sizes:
            overlap = max(1, int(tile_size * args.overlap_frac))
            overlap = min(overlap, tile_size - 1)

            # --- 3. First Inference ---
            prob_map_processed = tile_inference(
                model,
                process_image,
                tile_size=tile_size,
                overlap=overlap,
                device=device,
                num_classes=classes,
                use_gray=args.gray,
            )
            
            # --- 4. Orientation Check (Upright vs Upside Down) ---
            # Check wave distribution (Label 2) in Top vs Bottom half
            if classes > 2: # Need wave class
                pred_temp = np.argmax(prob_map_processed, axis=-1)
                ph, pw = pred_temp.shape
                mid_y = ph // 2
                # Count wave pixels
                top_waves = np.sum(pred_temp[:mid_y, :] == 2)
                bot_waves = np.sum(pred_temp[mid_y:, :] == 2)
                
                # Logic: Waveforms (Rhythm strips) are usually at the bottom.
                # If Top has significantly MORE waves than Bottom, or if the distribution is suspicious, check.
                # Wait, standard 12-lead has waves everywhere. But Rhythm strip is Long Lead II at bottom.
                # If Upside Down, the "Rhythm Strip" might be at the top? 
                # Or simply: Text headers are at top (BG), Rhythm at bottom (Wave).
                # Actually, 12-lead is dense waves. But usually bottom row is continuous.
                # Let's use user's logic: "看哪个区域mask2更多" (See which area has more Mask 2).
                # User says: "横着的肯定只有两种情况... 看哪个区域mask2更多"
                # Usually ECG has dense waves. If upside down, the "dense" part might be top?
                # Actually bottom rhythm strip is usually the densest horizontally (full width).
                # Top rows are split columns.
                # So Bottom Half usually has MORE Wave pixels than Top Half?
                # Calculating...
                # Top: 3 rows of short leads.
                # Bottom: 1 row of long lead.
                # Pixel count might be similar.
                # Let's look at DISTRIBUTION.
                # User suggestion: "看哪个区域mask2更多".
                # If Top > Bottom => Upside Down? (Assuming bottom should have more?)
                # Or does user imply "Rhythm strip is at bottom, so bottom should have waves"?
                # Let's assume correct orientation has MORE waves at bottom (due to long strip).
                # So if Top > Bottom * 1.2, it's likely Upside Down.
                
                ratio = top_waves / (bot_waves + 1e-5)
                print(f"[INFO] Orientation Check: Top Wave={top_waves}, Bot Wave={bot_waves}, Ratio={ratio:.2f}")
                
                if ratio > 1.2: # Tune this threshold
                    print(f"[INFO] 检测到倒置 (Top > Bot)，正在旋转180度重试...")
                    # Rotate Original 180
                    image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_180)
                    # Rotate Process 180 (no need to resize again, just rotate the smaller one)
                    process_image = cv2.rotate(process_image, cv2.ROTATE_180)
                    
                    # Re-Inference
                    prob_map_processed = tile_inference(
                        model,
                        process_image,
                        tile_size=tile_size,
                        overlap=overlap,
                        device=device,
                        num_classes=classes,
                        use_gray=args.gray,
                    )
            
            # --- 5. Resize result back ---

            
            # Resize back if needed
            if resize_scale != 1.0:
                 prob_map_all = cv2.resize(prob_map_processed, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                 # Ensure we have correct shape if resize squeezed dimensions (rare for linear but good to check)
                 if prob_map_all.ndim == 2:
                     prob_map_all = prob_map_all[..., None]
            else:
                 prob_map_all = prob_map_processed
            
            # Save raw probability maps? Maybe too big. 
            # Let's save the final prediction index map.
            if classes > 1:
                # Multi-class: argmax
                pred_map = np.argmax(prob_map_all, axis=-1) # (H, W)
                
                # Overlay
                overlay_img = overlay_classes(image_rgb, pred_map)
                
                # Save visualization
                cv2.imwrite(
                    str(output_dir / f"{path.stem}_t{tile_size}_overlay.png"),
                    cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR),
                )
                
                # Save prediction index mask (0, 1, 2) properly scaled for visibility or raw
                # For visibility: 0->0, 1->127, 2->255? Or just raw indices
                # Let's save colored mask for verify
                colored_mask = np.zeros_like(image_rgb)
                colored_mask[pred_map == 1] = [255, 0, 0] # Red Grid
                colored_mask[pred_map == 2] = [0, 255, 0] # Green Wave
                cv2.imwrite(
                    str(output_dir / f"{path.stem}_t{tile_size}_mask_vis.png"),
                    cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR),
                )
                
            else:
                # Binary legacy
                prob_map = prob_map_all[..., 0]
                mask = prob_map >= args.threshold
                overlay_img = overlay_classes(image_rgb, mask.astype(int), alpha=0.4) # Use same overlay logic logic roughly?
                # Actually, legacy logic was cleaner for binary
                overlay = image_rgb.copy()
                overlay[mask] = (overlay[mask] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
                
                cv2.imwrite(
                    str(output_dir / f"{path.stem}_t{tile_size}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )

            print(
                f"[INFO] {path.name} | tile {tile_size} | overlap {overlap}px | 已完成"
            )


if __name__ == "__main__":
    main()
