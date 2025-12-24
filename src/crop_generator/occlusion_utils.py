from __future__ import annotations

import random

import cv2
import numpy as np

from .config import OcclusionParams


def _choose_range(range_tuple):
    low, high = range_tuple
    if low > high:
        low, high = high, low
    return low, high


def apply_occlusions(image: np.ndarray, params: OcclusionParams) -> np.ndarray:
    if not params.enabled:
        return image

    height, width = image.shape[:2]
    short_side = max(1, min(height, width))

    low_count, high_count = _choose_range(params.count_range)
    count = random.randint(max(1, int(low_count)), max(1, int(high_count)))

    min_size_ratio, max_size_ratio = _choose_range(params.size_range)
    min_size = max(1, int(short_side * min_size_ratio))
    max_size = max(min_size, int(short_side * max_size_ratio))

    min_intensity, max_intensity = _choose_range(params.intensity_range)
    min_intensity = int(max(0, min_intensity))
    max_intensity = int(min(255, max_intensity))

    output = image.copy()
    if output.ndim == 2:
        tile_channels = 1
    else:
        tile_channels = output.shape[2]

    for _ in range(count):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        max_x = max(0, width - w)
        max_y = max(0, height - h)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x1 >= x2 or y1 >= y2:
            continue

        patch = output[y1:y2, x1:x2].astype(np.float32)

        # generate high-contrast noise block
        noise = np.random.rand(y2 - y1, x2 - x1).astype(np.float32)
        sigma = max(1.5, 0.1 * min(w, h))
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
        noise -= noise.min()
        denom = noise.max() - noise.min()
        if denom > 1e-6:
            noise /= denom

        intensity = random.randint(min_intensity, max_intensity)
        noise = (noise * 255.0).astype(np.float32)
        if intensity <= 128:
            base_block = np.clip(noise * random.uniform(0.25, 0.45), 0, random.uniform(60, 100))
        else:
            base_block = 255.0 - np.clip(noise * random.uniform(0.25, 0.45), 0, random.uniform(60, 100))

        rough_noise = np.random.normal(0, 40, size=base_block.shape).astype(np.float32)
        base_block += rough_noise
        base_block = np.clip(base_block, 0, 255)

        alpha = np.full(base_block.shape, random.uniform(0.85, 0.97), dtype=np.float32)
        if random.random() < 0.3:
            edge = cv2.GaussianBlur(np.ones_like(alpha), (0, 0), sigmaX=0.3 * min(w, h))
            edge -= edge.min()
            d = edge.max() - edge.min()
            if d > 1e-6:
                edge /= d
            alpha *= edge
            alpha = np.clip(alpha, 0.6, 0.98)

        if tile_channels == 1:
            patch = patch * (1.0 - alpha) + base_block * alpha
            output[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(output.dtype)
        else:
            blend_src = np.repeat(base_block[:, :, None], tile_channels, axis=2)
            patch = patch * (1.0 - alpha[..., None]) + blend_src * alpha[..., None]
            output[y1:y2, x1:x2] = np.clip(patch, 0, 255).astype(output.dtype)

    return output
