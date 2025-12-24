#!/usr/bin/env python3
"""
Batch-generate augmented ECG samples from a folder of original recordings.

For each source image, we produce a configurable number of 512x512 crops,
randomising the augmentation modules (wrinkles, blur, noise, occlusion, stains,
moiré, grayscale) and their parameters.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crop_generator import (  # noqa: E402
    BlurParams,
    CropParams,
    GrayParams,
    MoireParams,
    NegativeSampleParams,
    NoiseParams,
    OcclusionParams,
    StainParams,
    TransformParams,
    WrinkleParams,
    generate_negative_sample,
    generate_transformed_crops,
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


def iter_source_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.png")):
        if path.is_file():
            yield path


def generate_samples_for_image(
    image_path: Path,
    mask_path: Path,
    output_root: Path,
    samples_per_image: int,
    negative_per_image: int,
    negative_params: NegativeSampleParams,
    *,
    enable_wrinkle: bool,
    enable_blur: bool,
    force_blur: bool,
    enable_noise: bool,
    enable_occlusion: bool,
    enable_stain: bool,
    enable_moire: bool,
    enable_gray: bool,
    force_moire: bool,
    ensure_active_module: bool,
    rng: random.Random,
) -> None:
    image_id = image_path.stem
    target_dir = output_root / image_id
    target_dir.mkdir(parents=True, exist_ok=True)

    base_crop = CropParams(
        count=1,
        min_size=256,
        max_size=768,
        out_size=512,
        allow_out_of_bounds=False,
        corner_focus_prob=0.0,
        min_mask_coverage=0.5,
        coverage_dilation_kernel=9,
        coverage_max_attempts=60,
        write_metadata=True,
        min_horizontal_density=0.1,
        min_horizontal_rows=2,
    )

    corner_candidates = (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    )

    for idx in range(samples_per_image):
        transform = random_transform(rng)
        corner_override = corner_candidates[idx] if idx < len(corner_candidates) else None
        if corner_override is None:
            allow_oob = rng.random() < 0.4
        else:
            allow_oob = False
        crop_params = replace(base_crop, allow_out_of_bounds=allow_oob, corner_override=corner_override)

        modules = [
            random_wrinkle(rng) if enable_wrinkle else WrinkleParams(enabled=False),
            random_blur(rng, force=force_blur) if enable_blur else BlurParams(enabled=False),
            random_noise(rng) if enable_noise else NoiseParams(enabled=False),
            random_occlusion(rng) if enable_occlusion else OcclusionParams(enabled=False),
            random_stain(rng) if enable_stain else StainParams(enabled=False),
            random_moire(rng, force=force_moire) if enable_moire else MoireParams(enabled=False),
            random_gray(rng) if enable_gray else GrayParams(enabled=False),
        ]
        if ensure_active_module:
            ensure_any_module(modules, rng)

        wrinkle, blur, noise, occlusion, stain, moire, gray = modules

        sample_dir = target_dir / f"sample_{idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        seed = rng.randrange(0, 2**31)
        generate_transformed_crops(
            image_path=image_path,
            mask_path=mask_path,
            output_dir=sample_dir,
            transform=transform,
            crop=crop_params,
            wrinkle=wrinkle,
            blur=blur,
            gray=gray,
            noise=noise,
            occlusion=occlusion,
            stain=stain,
            moire=moire,
            seed=seed,
        )

        for src_name, dst_name in (("crop_00_img.png", "image.png"), ("crop_00_mask.png", "mask.png")):
            src = sample_dir / src_name
            if src.exists():
                src.rename(sample_dir / dst_name)

    for neg_idx in range(negative_per_image):
        neg_dir = target_dir / f"neg_{neg_idx:02d}"
        generate_negative_sample(output_dir=neg_dir, params=negative_params, rng=rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("ori_data"), help="Folder containing original ECG PNGs.")
    parser.add_argument("--mask", type=Path, default=Path("image_data/complete_mask.png"), help="Path to grid mask PNG.")
    parser.add_argument("--output-root", type=Path, default=Path("generated_samples"), help="Where to write augmented samples.")
    parser.add_argument("--samples-per-image", type=int, default=10, help="Number of augmented crops per source image.")
    parser.add_argument(
        "--negative-per-image",
        type=int,
        default=0,
        help="Number of synthetic negative samples (blank masks) per source image.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process this many source images.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility.")
    parser.add_argument(
        "--module-set",
        choices=["all", "moire-required", "moire-blur-required", "moire-only"],
        default="all",
        help=(
            "Which augmentation modules to enable. "
            "'moire-required' forces moiré while keeping other modules, "
            "'moire-blur-required' forces moiré and blur, "
            "'moire-only' keeps only the moiré stage."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if args.samples_per_image > 0 and not args.mask.exists():
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    source_images = list(iter_source_images(args.input_root))
    if args.limit is not None:
        source_images = source_images[: args.limit]

    if not source_images:
        raise RuntimeError(f"No PNG images found under {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    negative_params = NegativeSampleParams(image_size=512)

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

    for image_path in source_images:
        generate_samples_for_image(
            image_path=image_path,
            mask_path=args.mask,
            output_root=args.output_root,
            samples_per_image=args.samples_per_image,
            negative_per_image=args.negative_per_image,
            negative_params=negative_params,
            enable_wrinkle=enable_wrinkle,
            enable_blur=enable_blur,
            force_blur=force_blur,
            enable_noise=enable_noise,
            enable_occlusion=enable_occlusion,
            enable_stain=enable_stain,
            enable_moire=enable_moire,
            enable_gray=enable_gray,
            force_moire=force_moire,
            ensure_active_module=ensure_active_module,
            rng=rng,
        )


if __name__ == "__main__":
    main()
