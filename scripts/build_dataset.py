#!/usr/bin/env python3
"""
批量生成符合 datasets.GridMaskDataset 期望目录结构的数据集。

输出目录结构：
target_root/
  images/<sample>.png
  masks/<sample>_mask.png
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Allow importing from valid directory
sys.path.append(str(Path(__file__).parent))
from viz_dataset_masks import visualize_masks

from crop_generator import (  # noqa: E402
    BlurParams,
    ColorAugParams,
    CropParams,
    GrayParams,
    MoireParams,
    NegativeSampleParams,
    NoiseParams,
    OcclusionParams,
    StainParams,
    TextOverlayParams,
    TransformParams,
    WrinkleParams,
    generate_transformed_crops,
    synthesize_negative_image,
)
from crop_generator.random_params import (  # noqa: E402
    ensure_any_module,
    random_blur,
    random_gray,
    random_moire,
    random_noise,
    random_occlusion,
    random_stain,
    random_transform,
    random_wrinkle,
)


@dataclass(frozen=True)
class Job:
    index: int
    image_path: Path
    mask_path: Path
    coverage_mask_path: Path | None = None
    crop_overrides: dict[str, object] | None = None


@dataclass(frozen=True)
class WorkerConfig:
    samples_per_image: int
    negative_per_image: int
    allow_out_of_bounds_prob: float
    crop_min_size: int
    crop_max_size: int
    crop_out_size: int
    negative_params: NegativeSampleParams
    output_images: Path
    output_masks: Path
    tmp_root: Path
    overwrite: bool
    seed_base: int
    enable_wrinkle: bool
    enable_blur: bool
    force_blur: bool
    enable_noise: bool
    enable_occlusion: bool
    enable_stain: bool
    enable_moire: bool
    enable_gray: bool
    force_moire: bool
    ensure_active_module: bool


_WORKER_CONFIG: WorkerConfig | None = None


def _init_worker(config: WorkerConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    config.tmp_root.mkdir(parents=True, exist_ok=True)


def _iter_source_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.png")):
        if path.is_file():
            yield path


def _choose_modules(
    rng: random.Random,
    config: WorkerConfig,
) -> tuple[TransformParams, WrinkleParams, BlurParams, NoiseParams, OcclusionParams, StainParams, MoireParams, GrayParams]:
    transform = random_transform(rng)
    modules = []

    if config.enable_wrinkle:
        wrinkle = random_wrinkle(rng)
    else:
        wrinkle = WrinkleParams(enabled=False)
    modules.append(wrinkle)

    if config.enable_blur:
        blur = random_blur(rng, force=config.force_blur)
    else:
        blur = BlurParams(enabled=False)
    modules.append(blur)

    if config.enable_noise:
        noise = random_noise(rng)
    else:
        noise = NoiseParams(enabled=False)
    modules.append(noise)

    if config.enable_occlusion:
        occlusion = random_occlusion(rng)
    else:
        occlusion = OcclusionParams(enabled=False)
    modules.append(occlusion)

    if config.enable_stain:
        stain = random_stain(rng)
    else:
        stain = StainParams(enabled=False)
    modules.append(stain)

    if config.enable_moire:
        moire = random_moire(rng, force=config.force_moire)
    else:
        moire = MoireParams(enabled=False)
    modules.append(moire)

    if config.enable_gray:
        gray = random_gray(rng)
    else:
        gray = GrayParams(enabled=False)
    modules.append(gray)

    if config.ensure_active_module:
        ensure_any_module(modules, rng)

    wrinkle, blur, noise, occlusion, stain, moire, gray = modules[0], modules[1], modules[2], modules[3], modules[4], modules[5], modules[6]
    return transform, wrinkle, blur, noise, occlusion, stain, moire, gray


def _save_positive_sample(
    job: Job,
    sample_idx: int,
    transform: TransformParams,
    wrinkle: WrinkleParams,
    blur: BlurParams,
    noise: NoiseParams,
    occlusion: OcclusionParams,
    stain: StainParams,
    moire: MoireParams,
    gray: GrayParams,
    rng: random.Random,
) -> None:
    assert _WORKER_CONFIG is not None
    config = _WORKER_CONFIG
    mask_path = job.mask_path
    if mask_path is None:
        raise RuntimeError("未提供掩码路径，无法生成正样本。")

    stem = f"{job.image_path.stem}_p{sample_idx:02d}"
    final_image_path = config.output_images / f"{stem}.png"
    final_mask_path = config.output_masks / f"{stem}_mask.png"
    if not config.overwrite and (final_image_path.exists() or final_mask_path.exists()):
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="cg_", dir=config.tmp_root))
    try:
        corner_candidates = (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        )
        corner_override = corner_candidates[sample_idx] if sample_idx < len(corner_candidates) else None
        allow_oob = rng.random() < config.allow_out_of_bounds_prob if corner_override is None else False

        crop_params = CropParams(
            count=1,
            min_size=config.crop_min_size,
            max_size=config.crop_max_size,
            out_size=config.crop_out_size,
            allow_out_of_bounds=allow_oob,
            corner_override=corner_override,
            corner_focus_prob=0.0,
            min_mask_coverage=0.1,  # Relaxed from 0.3
            coverage_dilation_kernel=9,
            coverage_max_attempts=200,    # Increased from 60
            min_horizontal_density=0.05,  # Relaxed from 0.1
            min_horizontal_rows=1,        # Relaxed from 2
            mask_guided_prob=0.8,         # Actively search for waveforms
        )
        if job.crop_overrides:
            for key, value in job.crop_overrides.items():
                if hasattr(crop_params, key):
                    setattr(crop_params, key, value)
                else:
                    raise AttributeError(f"CropParams 不支持属性 {key}")
                    
        # Color Augmentation
        color_aug = ColorAugParams(
             enabled=True,
             brightness_range=(0.6, 1.15),
             contrast_range=(0.7, 1.3),
             saturation_range=(0.3, 1.5),
             hue_range=(-0.05, 0.05),
             warmth_range=(0.7, 1.3),
        )

        # Text Overlay Distractors
        text_overlay = TextOverlayParams(
            enabled=True,
            count_range=(2, 6),
            font_scale_range=(0.6, 1.4),
            opacity_range=(0.6, 0.95),
            clear_mask=False
        )

        try:
            generate_transformed_crops(
                image_path=job.image_path,
                mask_path=mask_path,
                coverage_mask_path=job.coverage_mask_path,
                output_dir=tmp_dir,
                transform=transform,
                crop=crop_params,
                wrinkle=wrinkle,
                blur=blur,
                gray=gray,
                noise=noise,
                occlusion=occlusion,
                stain=stain,
                moire=moire,
                color_aug=color_aug,
                text_overlay=text_overlay,
                seed=rng.randrange(0, 2**31),
            )
        except RuntimeError as e:
            # Skip this sample if generation fails (e.g. no valid crop found)
            print(f"[Warning] Skipping sample {stem} due to error: {e}")
            return

        src_image = tmp_dir / "crop_00_img.png"
        src_mask = tmp_dir / "crop_00_mask.png"
        if not src_image.exists() or not src_mask.exists():
            print(f"[Warning] Skipping sample {stem}: Output files not found after generation.")
            return

        config.output_images.mkdir(parents=True, exist_ok=True)
        config.output_masks.mkdir(parents=True, exist_ok=True)
        shutil.move(src=str(src_image), dst=final_image_path)
        shutil.move(src=str(src_mask), dst=final_mask_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _save_negative_sample(job: Job, neg_idx: int, rng: random.Random) -> None:
    assert _WORKER_CONFIG is not None
    config = _WORKER_CONFIG
    params = config.negative_params
    stem = f"{job.image_path.stem}_n{neg_idx:02d}"
    final_image_path = config.output_images / f"{stem}.png"
    final_mask_path = config.output_masks / f"{stem}_mask.png"
    if not config.overwrite and (final_image_path.exists() or final_mask_path.exists()):
        return

    image = synthesize_negative_image(params, rng)
    mask = np.zeros((params.image_size, params.image_size), dtype=np.uint8)

    config.output_images.mkdir(parents=True, exist_ok=True)
    config.output_masks.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(final_image_path)
    Image.fromarray(mask).save(final_mask_path)


def _process_job(job: Job) -> tuple[Path, int, int]:
    assert _WORKER_CONFIG is not None
    config = _WORKER_CONFIG
    rng = random.Random(config.seed_base + job.index)

    positive = 0
    for idx in range(config.samples_per_image):
        transform, wrinkle, blur, noise, occlusion, stain, moire, gray = _choose_modules(rng, config)
        _save_positive_sample(
            job,
            idx,
            transform,
            wrinkle,
            blur,
            noise,
            occlusion,
            stain,
            moire,
            gray,
            rng,
        )
        positive += 1

    negative = 0
    for idx in range(config.negative_per_image):
        _save_negative_sample(job, idx, rng)
        negative += 1
    return job.image_path, positive, negative


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("ori_data"), help="原始 ECG PNG 根目录。")
    parser.add_argument("--mask", type=Path, default=Path("image_data/complete_mask.png"), help="全局掩码图。")
    parser.add_argument(
        "--wave-root",
        type=Path,
        default=None,
        help="可选：包含波形整页 PNG 的根目录（会递归搜索 PNG 并使用对应 _mask.png 作为波形掩码）。",
    )
    parser.add_argument(
        "--wave-min-coverage",
        type=float,
        default=0.05,
        help="波形掩码 coverage 阈值（裁剪区域内波形像素占比下限，基于 dilate 后的掩码）。",
    )
    parser.add_argument(
        "--wave-coverage-kernel",
        type=int,
        default=45,
        help="波形掩码 coverage 计算时使用的膨胀核大小（像素）。",
    )
    parser.add_argument("--output-root", type=Path, default=Path("dataset_grid"), help="输出数据集根目录。")
    parser.add_argument("--samples-per-image", type=int, default=10, help="每张原图生成的正样本数量。")
    parser.add_argument("--negative-per-image", type=int, default=2, help="每张原图生成的负样本数量。")
    parser.add_argument("--allow-out-of-bounds-prob", type=float, default=0.4, help="裁剪允许越界的概率。")
    parser.add_argument("--crop-min", type=int, default=256, help="随机裁剪下限。")
    parser.add_argument("--crop-max", type=int, default=512, help="随机裁剪上限。")
    parser.add_argument("--crop-out", type=int, default=512, help="输出分辨率。")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 张原图（调试用）。")
    parser.add_argument("--seed", type=int, default=1337, help="随机种子。")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1), help="并行进程数。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出。")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="中间结果临时目录，默认为 output-root/_tmp。")
    parser.add_argument(
        "--module-set",
        choices=["all", "moire-required", "moire-blur-required", "moire-only"],
        default="all",
        help=(
            "选择启用的增强模块；'moire-required' 强制包含摩尔纹，"
            "'moire-blur-required' 同时强制摩尔纹与高斯模糊，'moire-only' 仅保留摩尔纹。"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    base_images: List[Path] = []
    if args.input_root.exists():
        base_images = list(_iter_source_images(args.input_root))
    else:
        print(f"警告：未找到输入目录 {args.input_root}，将跳过基础网格数据。")

    wave_images: List[Path] = []
    if args.wave_root:
        if args.wave_root.exists():
            wave_images = list(_iter_source_images(args.wave_root))
        else:
            print(f"警告：未找到波形目录 {args.wave_root}，将忽略 wave 数据。")

    sources: List[tuple[Path, Path, Path | None, dict[str, object] | None]] = []
    for path in base_images:
        sources.append((path, args.mask, None, None))

    if wave_images:
        wave_overrides: dict[str, object] = {
            "min_horizontal_density": 0.0,
            "min_horizontal_rows": 0,
        }
        if args.wave_min_coverage is not None:
            wave_overrides["min_mask_coverage"] = max(0.0, args.wave_min_coverage)
        if args.wave_coverage_kernel is not None and args.wave_coverage_kernel > 0:
            wave_overrides["coverage_dilation_kernel"] = int(args.wave_coverage_kernel)

        for path in wave_images:
            if path.stem.endswith("_mask"):
                continue
            coverage_mask = path.with_name(f"{path.stem}_mask{path.suffix}")
            if not coverage_mask.exists():
                print(f"警告：{coverage_mask} 不存在，跳过波形样本 {path.name}")
                continue
            sources.append((path, args.mask, coverage_mask, wave_overrides.copy()))

    if not sources:
        raise RuntimeError("未找到可用的输入 PNG，请检查 --input-root 或 --wave-root。")

    if args.limit is not None:
        sources = sources[: args.limit]

    images_dir = args.output_root / "images"
    masks_dir = args.output_root / "masks"
    tmp_root = args.tmp_dir or (args.output_root / "_tmp")

    if args.output_root.exists() and not args.overwrite:
        existing = list(args.output_root.glob("images/*.png"))
        if existing:
            raise FileExistsError(f"{args.output_root} 已包含数据，请使用 --overwrite 或清理目录。")

    if args.samples_per_image > 0 and not args.mask.exists():
        raise FileNotFoundError(f"未找到掩码文件：{args.mask}")

    negative_params = NegativeSampleParams(image_size=args.crop_out)

    if args.module_set == "all":
        enable_wrinkle = True
        enable_blur = True
        force_blur = False
        enable_noise = True
        enable_occlusion = True
        enable_stain = True
        enable_moire = True
        enable_gray = True
        force_moire = False
        ensure_active_module = True
    elif args.module_set == "moire-required":
        enable_wrinkle = True
        enable_blur = True
        force_blur = False
        enable_noise = True
        enable_occlusion = True
        enable_stain = True
        enable_moire = True
        enable_gray = True
        force_moire = True
        ensure_active_module = True
    elif args.module_set == "moire-blur-required":
        enable_wrinkle = True
        enable_blur = True
        force_blur = True
        enable_noise = True
        enable_occlusion = True
        enable_stain = True
        enable_moire = True
        enable_gray = True
        force_moire = True
        ensure_active_module = True
    else:  # moire-only
        enable_wrinkle = False
        enable_blur = False
        force_blur = False
        enable_noise = False
        enable_occlusion = False
        enable_stain = False
        enable_moire = True
        enable_gray = False
        force_moire = True
        ensure_active_module = False

    worker_config = WorkerConfig(
        samples_per_image=args.samples_per_image,
        negative_per_image=args.negative_per_image,
        allow_out_of_bounds_prob=args.allow_out_of_bounds_prob,
        crop_min_size=args.crop_min,
        crop_max_size=args.crop_max,
        crop_out_size=args.crop_out,
        negative_params=negative_params,
        output_images=images_dir,
        output_masks=masks_dir,
        tmp_root=tmp_root,
        overwrite=args.overwrite,
        seed_base=args.seed * 1_000_003,
        enable_wrinkle=enable_wrinkle,
        enable_blur=enable_blur,
        enable_noise=enable_noise,
        enable_occlusion=enable_occlusion,
        enable_stain=enable_stain,
        enable_moire=enable_moire,
        enable_gray=enable_gray,
        force_moire=force_moire,
        force_blur=force_blur,
        ensure_active_module=ensure_active_module,
    )

    jobs: list[Job] = []
    for idx, (image_path, mask_path, coverage_path, overrides) in enumerate(sources):
        jobs.append(
            Job(
                index=idx,
                image_path=image_path,
                mask_path=mask_path,
                coverage_mask_path=coverage_path,
                crop_overrides=overrides,
            )
        )

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(worker_config,)) as pool:
        progress = tqdm(total=len(jobs), desc="构建样本", unit="图", disable=len(jobs) == 0)
        try:
            for image_path, pos_count, neg_count in pool.imap_unordered(_process_job, jobs):
                print(f"[{image_path.name}] 正样本 {pos_count} 个，负样本 {neg_count} 个")
                progress.update(1)
        finally:
            progress.close()

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"完成，共处理 {len(jobs)} 张图片。输出目录：{args.output_root}")

    # Automatic Visualization
    viz_out = Path("viz_masks_output")
    if viz_out.exists():
        shutil.rmtree(viz_out)
    viz_out.mkdir(exist_ok=True)
    
    print("正在生成可视化样本 (10张)...")
    try:
        visualize_masks(
            mask_dir=args.output_root / "masks",
            output_dir=viz_out,
            count=10
        )
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")


if __name__ == "__main__":
    main()
