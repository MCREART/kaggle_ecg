#!/usr/bin/env python3
"""
CLI entry point for generating transformed ECG crops and 1px mask pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crop_generator import (  # noqa: E402
    BlurParams,
    CropParams,
    GrayParams,
    MoireParams,
    NoiseParams,
    OcclusionParams,
    PipelineConfig,
    StainParams,
    TransformParams,
    WrinkleParams,
    generate_transformed_crops,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="输入原始图像路径")
    parser.add_argument("--mask", required=True, type=Path, help="输入对应的二值网格 mask 路径")
    parser.add_argument("--output", required=True, type=Path, help="输出目录")
    parser.add_argument("--count", type=int, default=10, help="生成的裁剪对数量")
    parser.add_argument("--min-size", type=int, default=256, help="随机裁剪最小尺寸")
    parser.add_argument("--max-size", type=int, default=768, help="随机裁剪最大尺寸")
    parser.add_argument("--out-size", type=int, default=512, help="输出缩放尺寸（方形）")
    parser.add_argument(
        "--allow-out-of-bounds",
        action="store_true",
        help="允许裁剪窗口越过原始图像范围（越界部分将自动填充为黑色/0）",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=["none", "affine", "perspective"],
        default="perspective",
        help="所使用的几何变换类型",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子，设置后可复现结果")
    parser.add_argument("--max-rotate", type=float, default=8.0, help="仿射旋转最大角度（度）")
    parser.add_argument("--max-shift", type=float, default=0.08, help="仿射平移最大比例")
    parser.add_argument("--max-scale", type=float, default=0.10, help="仿射缩放最大比例")
    parser.add_argument("--perspective-jitter", type=float, default=0.12, help="透视四角偏移最大比例")
    parser.add_argument(
        "--wrinkles",
        action="store_true",
        help="启用纸张褶皱模拟（随机位移保持 mask 对齐）",
    )
    parser.add_argument(
        "--wrinkle-count-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="褶皱条数范围（默认 1-3 条）",
    )
    parser.add_argument(
        "--wrinkle-amplitude-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="褶皱位移幅度范围，相对于较短边的比例（默认 0.01-0.03）",
    )
    parser.add_argument(
        "--wrinkle-sigma-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="褶皱高斯宽度范围，相对于较短边的比例（默认 0.02-0.08）",
    )
    parser.add_argument(
        "--wrinkle-wavelength-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="褶皱正弦波波长范围，相对于较短边的比例（默认 0.08-0.18）",
    )
    parser.add_argument(
        "--blur",
        action="store_true",
        help="启用高斯模糊模拟相机/纸张失焦，仅作用于图像",
    )
    parser.add_argument(
        "--blur-kernel-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="模糊核尺寸范围（奇数，默认 3-7）",
    )
    parser.add_argument(
        "--blur-sigma-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="模糊 sigma 范围（默认 0.3-1.5）",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="启用灰度化处理，可模拟打印或扫描效果",
    )
    parser.add_argument(
        "--gray-single-channel",
        action="store_true",
        help="灰度化后输出单通道图像（默认保留 3 通道）",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="启用高斯噪声模拟扫描噪点",
    )
    parser.add_argument(
        "--noise-sigma-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="噪声标准差范围（默认 3-12）",
    )
    parser.add_argument(
        "--occlusion",
        action="store_true",
        help="启用随机矩形遮挡模拟折痕/手指遮挡等",
    )
    parser.add_argument(
        "--occlusion-count-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="遮挡块数量范围（默认 1-3）",
    )
    parser.add_argument(
        "--occlusion-size-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="遮挡块尺寸范围，占短边比例（默认 0.05-0.2）",
    )
    parser.add_argument(
        "--occlusion-intensity-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="遮挡块颜色范围（0-255，默认全范围）",
    )
    parser.add_argument(
        "--stain",
        action="store_true",
        help="启用深色污渍（高斯斑块）模拟纸面脏污",
    )
    parser.add_argument(
        "--stain-count-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="污渍数量范围（默认 1-3）",
    )
    parser.add_argument(
        "--stain-size-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="污渍尺寸范围，占短边比例（默认 0.05-0.15）",
    )
    parser.add_argument(
        "--stain-intensity-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="污渍加深强度（默认 35-120，越大越黑）",
    )
    parser.add_argument(
        "--stain-softness-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="污渍边缘柔和度（默认 0.6-1.2）",
    )
    parser.add_argument(
        "--stain-texture-strength-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="污渍纹理强度（默认 0.3-0.8，越大越斑驳）",
    )
    parser.add_argument(
        "--stain-texture-scale-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="纹理噪声模糊尺度（默认 0.05-0.25，相对于半径）",
    )
    parser.add_argument(
        "--stain-tint-rgb",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help="污渍颜色（RGB，默认不着色）。咖啡色推荐 150 110 60",
    )
    parser.add_argument(
        "--stain-tint-strength-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="污渍颜色混合强度（默认 0.2-0.6）",
    )
    parser.add_argument(
        "--moire",
        action="store_true",
        help="启用摩尔纹模拟（扫描/打印干涉条纹）",
    )
    parser.add_argument(
        "--moire-period-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="摩尔纹基准周期（像素，默认 6-18）",
    )
    parser.add_argument(
        "--moire-beat-ratio-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="第二频率相对比例（默认 1.02-1.15）",
    )
    parser.add_argument(
        "--moire-angle-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="摩尔纹方向角度范围（度，默认 -25~25）",
    )
    parser.add_argument(
        "--moire-amplitude-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="摩尔纹亮度调制强度（默认 0.05-0.18）",
    )
    return parser


def parse_args() -> PipelineConfig:
    parser = build_parser()
    args = parser.parse_args()

    transform_params = TransformParams(
        mode=args.transform,  # type: ignore[arg-type]
        max_rotate=args.max_rotate,
        max_shift=args.max_shift,
        max_scale=args.max_scale,
        perspective_jitter=args.perspective_jitter,
    )
    crop_params = CropParams(
        count=args.count,
        min_size=args.min_size,
        max_size=args.max_size,
        out_size=args.out_size,
        allow_out_of_bounds=args.allow_out_of_bounds,
    )

    wrinkle_params = WrinkleParams(enabled=args.wrinkles)
    if args.wrinkles:
        if args.wrinkle_count_range:
            wrinkle_params.count_range = tuple(args.wrinkle_count_range)  # type: ignore[assignment]
        if args.wrinkle_amplitude_range:
            wrinkle_params.amplitude_range = tuple(args.wrinkle_amplitude_range)  # type: ignore[assignment]
        if args.wrinkle_sigma_range:
            wrinkle_params.sigma_range = tuple(args.wrinkle_sigma_range)  # type: ignore[assignment]
        if args.wrinkle_wavelength_range:
            wrinkle_params.wavelength_range = tuple(args.wrinkle_wavelength_range)  # type: ignore[assignment]

    blur_params = BlurParams(enabled=args.blur)
    if args.blur:
        if args.blur_kernel_range:
            blur_params.kernel_range = tuple(args.blur_kernel_range)  # type: ignore[assignment]
        if args.blur_sigma_range:
            blur_params.sigma_range = tuple(args.blur_sigma_range)  # type: ignore[assignment]

    gray_params = GrayParams(enabled=args.grayscale)
    if args.grayscale and args.gray_single_channel:
        gray_params.preserve_channels = False

    noise_params = NoiseParams(enabled=args.noise)
    if args.noise and args.noise_sigma_range:
        noise_params.sigma_range = tuple(args.noise_sigma_range)  # type: ignore[assignment]

    occlusion_params = OcclusionParams(enabled=args.occlusion)
    if args.occlusion:
        if args.occlusion_count_range:
            occlusion_params.count_range = tuple(args.occlusion_count_range)  # type: ignore[assignment]
        if args.occlusion_size_range:
            occlusion_params.size_range = tuple(args.occlusion_size_range)  # type: ignore[assignment]
        if args.occlusion_intensity_range:
            occlusion_params.intensity_range = tuple(args.occlusion_intensity_range)  # type: ignore[assignment]

    stain_params = StainParams(enabled=args.stain)
    if args.stain:
        if args.stain_count_range:
            stain_params.count_range = tuple(args.stain_count_range)  # type: ignore[assignment]
        if args.stain_size_range:
            stain_params.size_range = tuple(args.stain_size_range)  # type: ignore[assignment]
        if args.stain_intensity_range:
            stain_params.intensity_range = tuple(args.stain_intensity_range)  # type: ignore[assignment]
        if args.stain_softness_range:
            stain_params.softness_range = tuple(args.stain_softness_range)  # type: ignore[assignment]
        if args.stain_texture_strength_range:
            stain_params.texture_strength_range = tuple(args.stain_texture_strength_range)  # type: ignore[assignment]
        if args.stain_texture_scale_range:
            stain_params.texture_scale_range = tuple(args.stain_texture_scale_range)  # type: ignore[assignment]
        if args.stain_tint_rgb:
            stain_params.tint_color = tuple(args.stain_tint_rgb)  # type: ignore[assignment]
        if args.stain_tint_strength_range:
            stain_params.tint_strength_range = tuple(args.stain_tint_strength_range)  # type: ignore[assignment]

    moire_params = MoireParams(enabled=args.moire)
    if args.moire:
        if args.moire_period_range:
            moire_params.period_range = tuple(args.moire_period_range)  # type: ignore[assignment]
        if args.moire_beat_ratio_range:
            moire_params.beat_ratio_range = tuple(args.moire_beat_ratio_range)  # type: ignore[assignment]
        if args.moire_angle_range:
            moire_params.angle_range = tuple(args.moire_angle_range)  # type: ignore[assignment]
        if args.moire_amplitude_range:
            moire_params.amplitude_range = tuple(args.moire_amplitude_range)  # type: ignore[assignment]

    return PipelineConfig(
        image_path=args.image,
        mask_path=args.mask,
        output_dir=args.output,
        seed=args.seed,
        transform=transform_params,
        crop=crop_params,
        wrinkle=wrinkle_params,
        blur=blur_params,
        gray=gray_params,
        noise=noise_params,
        occlusion=occlusion_params,
        stain=stain_params,
        moire=moire_params,
    )


def main() -> None:
    config = parse_args()
    generate_transformed_crops(
        config.image_path,
        config.mask_path,
        config.output_dir,
        transform=config.transform,
        crop=config.crop,
        wrinkle=config.wrinkle,
        blur=config.blur,
        gray=config.gray,
        noise=config.noise,
        occlusion=config.occlusion,
        stain=config.stain,
        moire=config.moire,
        seed=config.seed,
    )


if __name__ == "__main__":
    main()
