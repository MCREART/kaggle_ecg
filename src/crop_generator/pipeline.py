from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import json

from .config import (
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
)
from .blur_utils import apply_blur
from .grayscale_utils import apply_grayscale
from .mask_utils import resize_skeleton, zhang_suen_thinning
from .moire_utils import apply_moire
from .noise_utils import apply_noise
from .occlusion_utils import apply_occlusions
from .stain_utils import apply_stains
from .transformations import TransformResult, apply_transform
from .wrinkles import apply_wrinkles
from . import gpu_ops


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def load_inputs(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to load mask: {mask_path}")
    return image, mask


def _sample_edge_biased_coordinate(max_coord: int, edge_prob: float, span_ratio: float) -> int:
    if max_coord <= 0:
        return 0
    edge_prob = max(0.0, min(1.0, edge_prob))
    span_ratio = max(0.0, min(1.0, span_ratio))
    if edge_prob <= 0.0 or random.random() > edge_prob:
        return random.randint(0, max_coord)

    span = int(round(max_coord * span_ratio))
    span = max(1, min(span, max_coord))
    if random.random() < 0.5:
        return random.randint(0, span)
    start = max(0, max_coord - span)
    return random.randint(start, max_coord)


def _pad_with_random_border(image: np.ndarray, pad: int, noise_sigma: float) -> np.ndarray:
    if pad <= 0:
        return image

    dtype = image.dtype
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    if channels > 1:
        canvas = np.empty((height + 2 * pad, width + 2 * pad, channels), dtype=np.float32)
        base = np.random.randint(70, 210, size=channels)
        canvas[:] = base.reshape(1, 1, channels)
    else:
        canvas = np.empty((height + 2 * pad, width + 2 * pad), dtype=np.float32)
        base = float(np.random.randint(70, 210))
        canvas[:] = base

    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, size=canvas.shape).astype(np.float32)
        canvas += noise
        kernel = max(3, int(noise_sigma * 3) | 1)
        kernel = min(kernel, 31)
        if kernel % 2 == 0:
            kernel += 1
        canvas = cv2.GaussianBlur(
            canvas,
            (kernel, kernel),
            sigmaX=noise_sigma,
            sigmaY=noise_sigma,
            borderType=cv2.BORDER_REFLECT101,
        )

    if channels > 1:
        canvas[pad : pad + height, pad : pad + width, :] = image.astype(np.float32)
    else:
        canvas[pad : pad + height, pad : pad + width] = image.astype(np.float32)

    if np.issubdtype(dtype, np.floating):
        return canvas.astype(dtype)
    return np.clip(canvas, 0, 255).astype(dtype)


def _prepare_binary_mask(mask: np.ndarray) -> np.ndarray:
    mask_binary = (mask > 0).astype(np.uint8)
    if gpu_ops.available():
        thick = gpu_ops.binary_dilate(mask_binary, 3, iterations=1)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thick = cv2.dilate(mask_binary, kernel, iterations=1)
    return np.where(thick > 0, 255, 0).astype(np.uint8)


def _warp_additional_mask(mask: np.ndarray, transform: TransformResult) -> np.ndarray:
    height, width = mask.shape
    if transform.mode == "affine":
        warped = cv2.warpAffine(
            mask,
            transform.matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    elif transform.mode == "perspective":
        warped = cv2.warpPerspective(
            mask,
            transform.matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        warped = mask.copy()
    return warped


def _compute_mask_coverage(
    mask_binary: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    kernel: np.ndarray | None,
) -> float:
    region = mask_binary[y : y + height, x : x + width]
    if region.size == 0:
        return 0.0
    if kernel is not None:
        region = cv2.dilate(region, kernel)
    return float(region.mean())


def _sample_mask_guided_rectangle(
    mask_binary: np.ndarray,
    kernel: np.ndarray | None,
    crop_params: CropParams,
) -> tuple[int, int, int, int] | None:
    guided_mask = mask_binary
    if kernel is not None:
        guided_mask = cv2.dilate(mask_binary, kernel)

    min_area_ratio = max(0.0, crop_params.mask_guided_min_area_ratio)
    min_area_pixels = max(1, int(round(min_area_ratio * guided_mask.size)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(guided_mask, connectivity=8)
    regions: list[tuple[int, int, int, int, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        regions.append((area, x, y, w, h))

    if not regions:
        return None

    weights = [region[0] for region in regions]
    selected = random.choices(regions, weights=weights, k=1)[0]
    _, x, y, w, h = selected

    height, width = mask_binary.shape
    base_size = max(w, h)
    if base_size <= 0:
        return None

    expand_ratio = max(0.0, crop_params.mask_guided_expand_ratio)
    target_size = base_size * (1.0 + expand_ratio)
    target_size = max(float(crop_params.min_size), target_size)
    target_size = min(float(crop_params.max_size), float(min(width, height)))
    if target_size <= 0:
        return None

    jitter_ratio = max(0.0, crop_params.mask_guided_jitter_ratio)
    jitter_pixels = jitter_ratio * target_size
    center_x = x + w / 2.0 + random.uniform(-jitter_pixels, jitter_pixels)
    center_y = y + h / 2.0 + random.uniform(-jitter_pixels, jitter_pixels)

    rect_size = max(1, int(round(target_size)))
    rect_size = min(rect_size, width, height)
    half = rect_size / 2.0

    x0 = int(round(center_x - half))
    y0 = int(round(center_y - half))
    x0 = max(0, min(x0, width - rect_size))
    y0 = max(0, min(y0, height - rect_size))

    if width - rect_size < 0 or height - rect_size < 0:
        return None

    return x0, y0, rect_size, rect_size


def _is_uniform_region(region: np.ndarray, std_threshold: float, range_threshold: float) -> bool:
    if region.size == 0:
        return True
    std_threshold = max(0.0, std_threshold)
    range_threshold = max(0.0, range_threshold)
    if std_threshold <= 0 and range_threshold <= 0:
        return False

    region_f = region.astype(np.float32, copy=False)
    if region.ndim == 3:
        reshaped = region_f.reshape(-1, region.shape[2])
        std_val = float(np.max(reshaped.std(axis=0)))
        range_val = float(np.max(reshaped.max(axis=0) - reshaped.min(axis=0)))
    else:
        flat = region_f.reshape(-1)
        std_val = float(flat.std())
        range_val = float(flat.max() - flat.min())

    std_pass = std_threshold > 0 and std_val <= std_threshold
    range_pass = range_threshold > 0 and range_val <= range_threshold

    if std_threshold > 0 and range_threshold > 0:
        return std_pass and range_pass
    return std_pass or range_pass


def _generate_single_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_params: CropParams,
    index: int,
    output_dir: Path,
    coverage_mask: np.ndarray | None = None,
) -> None:
    height, width = mask.shape
    coverage_source = coverage_mask if coverage_mask is not None else mask
    mask_binary = (coverage_source > 0).astype(np.uint8)
    if mask_binary.sum() == 0:
        raise RuntimeError("裁剪前掩码全为零，无法生成有效样本")
    kernel = None
    if crop_params.coverage_dilation_kernel > 1:
        k = crop_params.coverage_dilation_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    min_coverage = max(0.0, crop_params.min_mask_coverage)
    max_attempts = max(1, crop_params.coverage_max_attempts) if min_coverage > 0 else 1

    def _sample_dimension(limit: int) -> int:
        limit = max(1, limit)
        lower = min(crop_params.min_size, limit)
        upper = min(crop_params.max_size, limit)
        if lower > upper:
            lower = upper
        return random.randint(lower, upper)

    corner_candidates = [
        (False, False),  # top-left
        (True, False),  # top-right
        (False, True),  # bottom-left
        (True, True),  # bottom-right
    ]
    can_corner = width >= crop_params.min_size and height >= crop_params.min_size
    forced_corner = crop_params.corner_focus_prob > 0.0 and index < len(corner_candidates) and can_corner
    override_corner = crop_params.corner_override if can_corner else None

    choose_x_upper = False
    choose_y_upper = False
    corner_focus = False

    if override_corner is not None:
        choose_x_upper, choose_y_upper = override_corner
        corner_focus = True
    else:
        if forced_corner:
            choose_x_upper, choose_y_upper = corner_candidates[index]
            corner_focus = True
        elif can_corner and random.random() < crop_params.corner_focus_prob:
            choose_x_upper = random.random() < 0.5
            choose_y_upper = random.random() < 0.5
            corner_focus = True

    def sample_rectangle() -> tuple[int, int, int, int]:
        if corner_focus:
            width_limit = width
            height_limit = height
            rect_width = _sample_dimension(width_limit)
            rect_height = _sample_dimension(height_limit)

            vertex_x = width - 1 if choose_x_upper else 0
            vertex_y = height - 1 if choose_y_upper else 0

            x0 = vertex_x - rect_width + 1 if choose_x_upper else vertex_x
            y0 = vertex_y - rect_height + 1 if choose_y_upper else vertex_y

            x0 = max(0, min(x0, width - rect_width))
            y0 = max(0, min(y0, height - rect_height))

            return x0, y0, rect_width, rect_height

        limit = min(width, height)
        rect_size = _sample_dimension(limit)
        rect_width = rect_height = min(rect_size, limit)
        max_x = max(0, width - rect_width)
        max_y = max(0, height - rect_height)
        return (
            _sample_edge_biased_coordinate(max_x, crop_params.edge_focus_prob, crop_params.edge_focus_span_ratio),
            _sample_edge_biased_coordinate(max_y, crop_params.edge_focus_prob, crop_params.edge_focus_span_ratio),
            rect_width,
            rect_height,
        )

    def evaluate(rect: tuple[int, int, int, int]) -> float:
        return _compute_mask_coverage(mask_binary, rect[0], rect[1], rect[2], rect[3], kernel)

    std_threshold = crop_params.uniform_std_threshold
    range_threshold = crop_params.uniform_range_threshold
    coverage_attempts = max(1, crop_params.coverage_max_attempts)
    outer_attempts = max(1, crop_params.uniform_retry_attempts)
    guided_enabled = crop_params.mask_guided_prob > 0.0 and random.random() < crop_params.mask_guided_prob

    best_rect: tuple[int, int, int, int] | None = None
    best_cov = -1.0

    for _ in range(outer_attempts):
        best_cov = -1.0
        best_rect = None
        best_non_uniform_rect: tuple[int, int, int, int] | None = None
        best_non_uniform_cov = -1.0
        best_any_rect: tuple[int, int, int, int] | None = None
        best_any_cov = -1.0

        for _ in range(coverage_attempts):
            if guided_enabled:
                rect = _sample_mask_guided_rectangle(mask_binary, kernel, crop_params)
                if rect is None:
                    guided_enabled = False
                    rect = sample_rectangle()
            else:
                rect = sample_rectangle()
            x, y, rect_width, rect_height = rect
            if rect_width <= 0 or rect_height <= 0:
                continue
            region = image[y : y + rect_height, x : x + rect_width]
            uniform = _is_uniform_region(region, std_threshold, range_threshold)
            cov = evaluate(rect)

            if cov > best_any_cov:
                best_any_cov = cov
                best_any_rect = rect

            if not uniform and cov > best_non_uniform_cov:
                best_non_uniform_cov = cov
                best_non_uniform_rect = rect

            if uniform:
                continue

            if cov > best_cov:
                best_cov = cov
                best_rect = rect
            if min_coverage <= 0.0 or cov >= min_coverage:
                break

        if best_rect is None:
            if best_non_uniform_rect is not None:
                best_rect = best_non_uniform_rect
                best_cov = best_non_uniform_cov
            elif best_any_rect is not None:
                best_rect = best_any_rect
                best_cov = best_any_cov

        if best_rect is None:
            continue

        x0, y0, w0, h0 = best_rect
        final_region = image[y0 : y0 + h0, x0 : x0 + w0]
        if _is_uniform_region(final_region, std_threshold, range_threshold):
            best_rect = None
            best_cov = -1.0
            continue

        if min_coverage > 0.0 and best_cov < min_coverage:
            if best_cov <= 0.0:
                best_rect = None
                best_cov = -1.0
                continue

        if crop_params.min_horizontal_rows > 0 and crop_params.min_horizontal_density > 0.0:
            region_mask = mask_binary[y0 : y0 + h0, x0 : x0 + w0]
            if region_mask.size == 0:
                best_rect = None
                best_cov = -1.0
                continue
            row_counts = region_mask.sum(axis=1)
            threshold = max(1, int(round(crop_params.min_horizontal_density * w0)))
            horizontal_rows = int(np.count_nonzero(row_counts >= threshold))
            if horizontal_rows < crop_params.min_horizontal_rows:
                best_rect = None
                best_cov = -1.0
                continue

        break

    if best_rect is None:
        raise RuntimeError("未能找到满足要求的裁剪区域，请检查掩码或调整参数")

    x, y, rect_width, rect_height = best_rect

    crop_img = image[y : y + rect_height, x : x + rect_width]
    crop_mask = mask[y : y + rect_height, x : x + rect_width]

    resized_img = cv2.resize(
        crop_img, (crop_params.out_size, crop_params.out_size), interpolation=cv2.INTER_LINEAR
    )

    # ... Helper function to process binary mask chunks ...
    def _process_chunk_mask(chunk_mask: np.ndarray, params: CropParams) -> np.ndarray:
        binary_chunk = (chunk_mask > 0).astype(np.uint8)
        if gpu_ops.available():
            closed = gpu_ops.binary_closing(binary_chunk, 3)
        else:
            k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(binary_chunk, cv2.MORPH_CLOSE, k_close)
        
        thinned = zhang_suen_thinning(closed)
        resized_skel = resize_skeleton(thinned, params.out_size)
        
        if gpu_ops.available():
            thick = gpu_ops.binary_dilate((resized_skel > 0).astype(np.uint8), 3, iterations=1)
            final_chunk = np.where(thick > 0, 1, 0).astype(np.uint8)
        else:
            k_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(resized_skel, k_thick, iterations=1)
            final_chunk = np.where(dilated > 0, 1, 0).astype(np.uint8)
        return final_chunk

    # Process Grid Mask (Label 1)
    grid_processed = _process_chunk_mask(crop_mask, crop_params)
    
    # Process Wave/Coverage Mask if available (Label 2)
    wave_processed = None
    if coverage_mask is not None:
        crop_wave = coverage_mask[y : y + rect_height, x : x + rect_width]
        wave_processed = _process_chunk_mask(crop_wave, crop_params)

    # Merge labels: 0=BG, 1=Grid, 2=Wave
    # Priority: Wave (2) > Grid (1) > BG (0)
    final_combined_mask = np.zeros((crop_params.out_size, crop_params.out_size), dtype=np.uint8)
    final_combined_mask[grid_processed > 0] = 1
    if wave_processed is not None:
        final_combined_mask[wave_processed > 0] = 2

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"crop_{index:02d}_img.png"
    mask_path = output_dir / f"crop_{index:02d}_mask.png"
    Image.fromarray(resized_img).save(image_path)
    Image.fromarray(final_combined_mask).save(mask_path)
    if crop_params.write_metadata:
        metadata = {
            "x": int(x),
            "y": int(y),
            "width": int(rect_width),
            "height": int(rect_height),
            "corner_override": crop_params.corner_override,
            "corner_focus": corner_focus,
            "allow_out_of_bounds": crop_params.allow_out_of_bounds,
        }
        meta_path = output_dir / f"crop_{index:02d}_meta.json"
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False, indent=2)


def run_pipeline(config: PipelineConfig) -> None:
    set_seed(config.seed)
    image, mask = load_inputs(config.image_path, config.mask_path)
    coverage_mask = None
    if config.coverage_mask_path is not None:
        coverage_mask = cv2.imread(str(config.coverage_mask_path), cv2.IMREAD_GRAYSCALE)
        if coverage_mask is None:
            raise FileNotFoundError(f"Unable to load coverage mask: {config.coverage_mask_path}")
        if coverage_mask.shape != mask.shape:
            raise ValueError("coverage_mask 与主掩码尺寸不一致")

    mask = _prepare_binary_mask(mask)
    if coverage_mask is not None:
        coverage_mask = _prepare_binary_mask(coverage_mask)

    transform_result = apply_transform(
        image=image,
        mask=mask,
        params=config.transform,
    )
    transformed_image = transform_result.image
    transformed_mask = transform_result.mask
    coverage_mask_transformed = None
    if coverage_mask is not None:
        coverage_mask_transformed = _warp_additional_mask(coverage_mask, transform_result)

    wrinkle_result = apply_wrinkles(
        transformed_image,
        transformed_mask,
        config.wrinkle,
        extra_mask=coverage_mask_transformed,
    )
    wrinkled_image = wrinkle_result.image
    wrinkled_mask = wrinkle_result.mask
    coverage_mask_wrinkled = wrinkle_result.extra_mask
    blurred_image = apply_blur(wrinkled_image, config.blur)
    noised_image = apply_noise(blurred_image, config.noise)
    occluded_image = apply_occlusions(noised_image, config.occlusion)
    stained_image = apply_stains(occluded_image, config.stain)
    moire_image = apply_moire(stained_image, config.moire)
    gray_image = apply_grayscale(moire_image, config.gray)

    image_for_crops = gray_image
    mask_for_crops = wrinkled_mask
    coverage_for_crops = coverage_mask_wrinkled
    if config.crop.allow_out_of_bounds:
        pad = config.crop.max_size
        if config.crop.random_border_fill:
            image_for_crops = _pad_with_random_border(image_for_crops, pad, config.crop.border_noise_sigma)
        else:
            pad_value = (0, 0, 0) if image_for_crops.ndim == 3 else 0
            image_for_crops = cv2.copyMakeBorder(
                image_for_crops,
                pad,
                pad,
                pad,
                pad,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_value,
            )
        mask_for_crops = cv2.copyMakeBorder(
            wrinkled_mask,
            pad,
            pad,
            pad,
            pad,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )
        if coverage_for_crops is not None:
            coverage_for_crops = cv2.copyMakeBorder(
                coverage_for_crops,
                pad,
                pad,
                pad,
                pad,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )

    for idx in range(config.crop.count):
        _generate_single_crop(
            image_for_crops,
            mask_for_crops,
            config.crop,
            idx,
            config.output_dir,
            coverage_mask=coverage_for_crops,
        )


def generate_transformed_crops(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    *,
    coverage_mask_path: Path | None = None,
    transform: TransformParams | None = None,
    crop: CropParams | None = None,
    wrinkle: WrinkleParams | None = None,
    blur: BlurParams | None = None,
    gray: GrayParams | None = None,
    moire: MoireParams | None = None,
    noise: NoiseParams | None = None,
    occlusion: OcclusionParams | None = None,
    stain: StainParams | None = None,
    seed: int | None = None,
) -> None:
    config = PipelineConfig(
        image_path=image_path,
        mask_path=mask_path,
        coverage_mask_path=coverage_mask_path,
        output_dir=output_dir,
        seed=seed,
        transform=transform or TransformParams(),
        crop=crop or CropParams(),
        wrinkle=wrinkle or WrinkleParams(),
        blur=blur or BlurParams(),
        gray=gray or GrayParams(),
        moire=moire or MoireParams(),
        noise=noise or NoiseParams(),
        occlusion=occlusion or OcclusionParams(),
        stain=stain or StainParams(),
    )
    retries = max(1, config.crop.global_retry_attempts)
    original_allow_oob = config.crop.allow_out_of_bounds
    base_seed = seed
    for attempt in range(retries):
        if base_seed is None:
            config.seed = None
        else:
            config.seed = base_seed + (attempt * 9973) + 1
        try:
            run_pipeline(config)
            config.crop.allow_out_of_bounds = original_allow_oob
            return
        except RuntimeError:
            if original_allow_oob and config.crop.allow_out_of_bounds:
                config.crop.allow_out_of_bounds = False
                continue
            if attempt == retries - 1:
                config.crop.allow_out_of_bounds = original_allow_oob
                raise
    config.crop.allow_out_of_bounds = original_allow_oob
