#!/usr/bin/env python3
"""
快速演示当前增强模块（褶皱、模糊、噪声、噪声遮挡、污渍、摩尔纹、灰度）的不同强度效果。

默认使用 image_data/1006427285-0001.png，先中心裁剪 512×512，再输出到 demo/ 目录：
  demo/
    wrinkles_light.png
    wrinkles_mid.png
    ...

可通过 --source / --output 指定其他路径。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crop_generator.blur_utils import apply_blur
from crop_generator.config import (
    BlurParams,
    GrayParams,
    MoireParams,
    NoiseParams,
    OcclusionParams,
    StainParams,
    WrinkleParams,
)
from crop_generator.grayscale_utils import apply_grayscale
from crop_generator.moire_utils import apply_moire
from crop_generator.noise_utils import apply_noise
from crop_generator.occlusion_utils import apply_occlusions
from crop_generator.stain_utils import apply_stains
from crop_generator.wrinkles import apply_wrinkles

SUMMARY_LINES: list[str] = []


def record_summary(module: str, level: str, detail: str) -> None:
    SUMMARY_LINES.append(f"{module} [{level}]: {detail}")


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def prepare_image(image: np.ndarray, target: int) -> np.ndarray:
    """中心裁剪成 target×target（不缩放），方便在编辑器中快速查看。"""
    h, w = image.shape[:2]
    if h < target or w < target:
        raise ValueError(f"输入图片尺寸 {w}x{h} 小于目标裁剪 {target}x{target}")
    top = max(0, (h - target) // 2)
    left = max(0, (w - target) // 2)
    return image[top : top + target, left : left + target]


def save_image(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def demo_wrinkles(image: np.ndarray, out_dir: Path) -> None:
    h, w = image.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    settings = {
        "min": (
            WrinkleParams(
                enabled=True,
                count_range=(1, 1),
                amplitude_range=(0.01, 0.012),
                sigma_range=(0.05, 0.06),
                wavelength_range=(0.12, 0.16),
            ),
            "count=1, amplitude=1%-1.2%短边, sigma=5%-6%, wavelength=12%-16%",
        ),
        "mid": (
            WrinkleParams(
                enabled=True,
                count_range=(2, 2),
                amplitude_range=(0.02, 0.03),
                sigma_range=(0.035, 0.045),
                wavelength_range=(0.09, 0.13),
            ),
            "count=2, amplitude=2%-3%, sigma≈4%, wavelength=9%-13%",
        ),
        "max": (
            WrinkleParams(
                enabled=True,
                count_range=(3, 4),
                amplitude_range=(0.04, 0.06),
                sigma_range=(0.02, 0.035),
                wavelength_range=(0.06, 0.1),
            ),
            "count=3-4, amplitude=4%-6%, sigma≈2%-3.5%, wavelength=6%-10%",
        ),
    }
    for label, (params, detail) in settings.items():
        result = apply_wrinkles(image, mask, params)
        save_image(result.image, out_dir / f"wrinkles_{label}.png")
        record_summary("wrinkles", label, detail)


def demo_blur(image: np.ndarray, out_dir: Path) -> None:
    settings = {
        "min": (
            BlurParams(enabled=True, kernel_range=(3, 3), sigma_range=(0.35, 0.4)),
            "kernel=3, sigma≈0.35-0.4 (接近真实最小模糊)",
        ),
        "mid": (
            BlurParams(enabled=True, kernel_range=(5, 5), sigma_range=(0.9, 1.1)),
            "kernel=5, sigma≈0.9-1.1",
        ),
        "max": (
            BlurParams(enabled=True, kernel_range=(9, 9), sigma_range=(1.5, 1.8)),
            "kernel=9, sigma≈1.5-1.8 (接近真实最大模糊)",
        ),
    }
    for label, (params, detail) in settings.items():
        save_image(apply_blur(image, params), out_dir / f"blur_{label}.png")
        record_summary("blur", label, detail)


def demo_noise(image: np.ndarray, out_dir: Path) -> None:
    settings = {
        "min": (
            NoiseParams(enabled=True, sigma_range=(3.0, 4.0)),
            "σ=3~4 (接近最小噪声)",
        ),
        "mid": (
            NoiseParams(enabled=True, sigma_range=(7.0, 8.5)),
            "σ=7~8.5",
        ),
        "max": (
            NoiseParams(enabled=True, sigma_range=(12.0, 14.0)),
            "σ=12~14 (接近最大噪声)",
        ),
    }
    for label, (params, detail) in settings.items():
        save_image(apply_noise(image, params), out_dir / f"noise_{label}.png")
        record_summary("noise", label, detail)


def demo_occlusion(image: np.ndarray, out_dir: Path) -> None:
    settings = {
        "min": (
            OcclusionParams(
                enabled=True,
                count_range=(1, 1),
                size_range=(0.08, 0.08),
                intensity_range=(110, 120),
            ),
            "count=1, size=8%短边, intensity=110-120 (浅色块)",
        ),
        "mid": (
            OcclusionParams(
                enabled=True,
                count_range=(2, 2),
                size_range=(0.15, 0.16),
                intensity_range=(40, 60),
            ),
            "count=2, size≈15%, intensity=40-60",
        ),
        "max": (
            OcclusionParams(
                enabled=True,
                count_range=(3, 4),
                size_range=(0.25, 0.28),
                intensity_range=(0, 20),
            ),
            "count=3-4, size=25%-28%, intensity=0-20 (极深色块)",
        ),
    }
    for idx, (label, (params, detail)) in enumerate(settings.items()):
        random.seed(100 + idx)
        save_image(apply_occlusions(image, params), out_dir / f"occlusion_{label}.png")
        record_summary("occlusion", label, detail)


def demo_stain(image: np.ndarray, out_dir: Path) -> None:
    settings = {
        "min": (
            StainParams(
                enabled=True,
                count_range=(1, 1),
                size_range=(0.05, 0.06),
                intensity_range=(20.0, 30.0),
                tint_color=(135, 110, 80),
                tint_strength_range=(0.15, 0.2),
            ),
            "count=1, 5%-6%短边, 变暗20-30, 浅咖色",
        ),
        "mid": (
            StainParams(
                enabled=True,
                count_range=(2, 3),
                size_range=(0.09, 0.12),
                intensity_range=(40.0, 70.0),
                tint_color=(115, 85, 55),
                tint_strength_range=(0.25, 0.35),
            ),
            "count=2-3, 9%-12%, 变暗40-70",
        ),
        "max": (
            StainParams(
                enabled=True,
                count_range=(3, 4),
                size_range=(0.15, 0.2),
                intensity_range=(70.0, 120.0),
                tint_color=(95, 60, 30),
                tint_strength_range=(0.35, 0.45),
            ),
            "count=3-4, 15%-20%, 变暗70-120 (重咖啡渍)",
        ),
    }
    for idx, (label, (params, detail)) in enumerate(settings.items()):
        random.seed(200 + idx)
        save_image(apply_stains(image, params), out_dir / f"stain_{label}.png")
        record_summary("stain", label, detail)


def demo_moire(image: np.ndarray, out_dir: Path) -> None:
    settings = {
        "min": (
            MoireParams(
                enabled=True,
                period_range=(12.0, 14.0),
                amplitude_range=(0.07, 0.09),
                blend_range=(0.88, 0.95),
                secondary_prob=0.1,
            ),
            "period=12-14px, amplitude=0.07-0.09, blend≈0.9",
        ),
        "mid": (
            MoireParams(
                enabled=True,
                period_range=(8.0, 10.0),
                amplitude_range=(0.15, 0.2),
                blend_range=(0.75, 0.85),
                secondary_prob=0.4,
            ),
            "period=8-10px, amplitude=0.15-0.2, blend≈0.8",
        ),
        "max": (
            MoireParams(
                enabled=True,
                period_range=(5.0, 6.5),
                amplitude_range=(0.25, 0.3),
                blend_range=(0.6, 0.7),
                secondary_prob=0.9,
                contrast_gain_range=(2.5, 3.0),
            ),
            "period=5-6.5px, amplitude=0.25-0.3, blend≈0.6-0.7, secondary≈90%",
        ),
    }
    for idx, (label, (params, detail)) in enumerate(settings.items()):
        random.seed(300 + idx)
        save_image(apply_moire(image, params), out_dir / f"moire_{label}.png")
        record_summary("moire", label, detail)


def demo_gray(image: np.ndarray, out_dir: Path) -> None:
    settings = {
        "preserve": GrayParams(enabled=True, preserve_channels=True),
        "single": GrayParams(enabled=True, preserve_channels=False),
    }
    for label, params in settings.items():
        save_image(apply_grayscale(image, params), out_dir / f"gray_{label}.png")


def dump_summary(output_dir: Path) -> None:
    if not SUMMARY_LINES:
        return
    summary_path = output_dir / "augmentation_levels.txt"
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write("各模块示例所用参数（近似对应真实合成的最小/中间/最大水平）\n")
        for line in SUMMARY_LINES:
            fp.write(f"- {line}\n")


def run_demo(source: Path, output_dir: Path, crop_size: int) -> None:
    image = load_image(source)
    image = prepare_image(image, crop_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    for png in output_dir.glob("*.png"):
        png.unlink()
    summary_file = output_dir / "augmentation_levels.txt"
    if summary_file.exists():
        summary_file.unlink()
    SUMMARY_LINES.clear()
    demo_wrinkles(image, output_dir)
    demo_blur(image, output_dir)
    demo_noise(image, output_dir)
    demo_occlusion(image, output_dir)
    demo_stain(image, output_dir)
    demo_moire(image, output_dir)
    demo_gray(image, output_dir)
    dump_summary(output_dir)
    print(f"Demo 图像已输出到 {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        default=Path("image_data/1006427285-0001.png"),
        help="演示所用的基础 PNG",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("demo"),
        help="输出目录（会自动创建）",
    )
    ap.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="演示前先中心裁剪到该尺寸（默认 512×512）",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.source, args.output, args.crop_size)
