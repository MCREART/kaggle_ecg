from __future__ import annotations

import random

import cv2
import numpy as np

from .config import StainParams


def _choose_range(range_tuple):
    low, high = range_tuple
    if low > high:
        low, high = high, low
    return low, high


def apply_stains(image: np.ndarray, params: StainParams) -> np.ndarray:
    if not params.enabled:
        return image

    height, width = image.shape[:2]
    short_side = max(1, min(height, width))

    min_count, max_count = _choose_range(params.count_range)
    count = random.randint(max(1, int(min_count)), max(1, int(max_count)))

    min_size_ratio, max_size_ratio = _choose_range(params.size_range)
    min_soft, max_soft = _choose_range(params.softness_range)
    min_intensity, max_intensity = _choose_range(params.intensity_range)
    min_tint, max_tint = _choose_range(params.tint_strength_range)
    min_texture, max_texture = _choose_range(params.texture_strength_range)
    min_scale, max_scale = _choose_range(params.texture_scale_range)
    tint_color = None
    if params.tint_color is not None:
        tint_color = np.array(params.tint_color, dtype=np.float32)

    output = image.astype(np.float32)

    for _ in range(count):
        radius = random.uniform(min_size_ratio, max_size_ratio) * short_side
        radius = max(3.0, radius)
        softness = random.uniform(min_soft, max_soft)
        sigma = max(1.5, radius * softness)

        amplitude = random.uniform(min_intensity, max_intensity)
        tint_strength = random.uniform(min_tint, max_tint) if tint_color is not None else 0.0
        texture_strength = random.uniform(min_texture, max_texture)
        texture_scale = random.uniform(min_scale, max_scale)

        cx = random.uniform(-0.1 * width, 1.1 * width)
        cy = random.uniform(-0.1 * height, 1.1 * height)

        x1 = int(max(0, cx - 3 * sigma))
        x2 = int(min(width, cx + 3 * sigma))
        y1 = int(max(0, cy - 3 * sigma))
        y2 = int(min(height, cy + 3 * sigma))

        if x1 >= x2 or y1 >= y2:
            continue

        x = np.arange(x1, x2, dtype=np.float32)
        y = np.arange(y1, y2, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        base = np.exp(-dist2 / (2.0 * sigma ** 2 + 1e-6))
        base = np.power(base, 0.7)

        h_patch = y2 - y1
        w_patch = x2 - x1
        noise = np.random.rand(h_patch, w_patch).astype(np.float32)
        sigma_noise = max(1.0, radius * texture_scale)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma_noise, sigmaY=sigma_noise)
        noise -= noise.min()
        denom = noise.max() - noise.min()
        if denom > 1e-6:
            noise /= denom
        texture = 1.0 + texture_strength * (noise - 0.5)
        gaussian = amplitude * base * texture

        if output.ndim == 3:
            patch = output[y1:y2, x1:x2, :]
            patch -= gaussian[..., None]
            if tint_color is not None and tint_strength > 0:
                blend = (gaussian[..., None] / 255.0) * tint_strength
                patch *= (1.0 - blend)
                patch += tint_color * blend
            output[y1:y2, x1:x2, :] = patch
        else:
            output[y1:y2, x1:x2] -= gaussian

    np.clip(output, 0, 255, out=output)
    return output.astype(image.dtype)
