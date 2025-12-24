#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from __future__ import annotations

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


def load_model(run_dir: Path, checkpoint: Path | None, device: torch.device) -> tuple[nn.Module, dict]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到 config.json: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
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

    prob_map = np.zeros((padded.shape[0], padded.shape[1]), dtype=np.float32)
    weight_map = np.zeros_like(prob_map)

    for y in range(0, padded.shape[0] - tile_size + 1, stride):
        for x in range(0, padded.shape[1] - tile_size + 1, stride):
            tile = padded[y : y + tile_size, x : x + tile_size]
            tensor = preprocess_image(tile).to(device)
            with torch.no_grad():
                logits = model(tensor)
                if logits.shape[1] == 1:
                    probs = torch.sigmoid(logits)
                    pred = probs.squeeze().cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=1)
                    pred = probs[:, 1].squeeze().cpu().numpy()

            prob_map[y : y + tile_size, x : x + tile_size] += pred
            weight_map[y : y + tile_size, x : x + tile_size] += 1.0

    weight_map[weight_map == 0] = 1.0
    prob_map /= weight_map
    return prob_map[:h, :w]


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay = image.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)
    overlay[mask] = (overlay[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_model(args.run_dir, args.checkpoint, device)

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

        for tile_size in args.tile_sizes:
            overlap = max(1, int(tile_size * args.overlap_frac))
            overlap = min(overlap, tile_size - 1)

            prob_map = tile_inference(
                model,
                image_rgb,
                tile_size=tile_size,
                overlap=overlap,
                device=device,
            )
            mask = prob_map >= args.threshold
            overlay_img = overlay_mask(image_rgb, mask)

            base = path.stem
            suffix = f"tile{tile_size}"
            cv2.imwrite(
                str(output_dir / f"{base}_{suffix}_prob.png"),
                (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(output_dir / f"{base}_{suffix}_mask.png"),
                mask.astype(np.uint8) * 255,
            )
            cv2.imwrite(
                str(output_dir / f"{base}_{suffix}_overlay.png"),
                cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR),
            )
            print(
                f"[INFO] {path.name} | tile {tile_size} | overlap {overlap}px | 已完成"
            )


if __name__ == "__main__":
    main()
