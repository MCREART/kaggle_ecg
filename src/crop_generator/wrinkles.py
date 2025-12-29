from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import WrinkleParams
from .mask_utils import zhang_suen_thinning
from .vector_mask import WrinkleWave


def _ensure_range(range_tuple: Tuple[float, float], min_value: float = 0.0) -> Tuple[float, float]:
    low, high = range_tuple
    if low > high:
        low, high = high, low
    low = max(low, min_value)
    high = max(high, min_value + 1e-6)
    return low, high


@dataclass(frozen=True)
class WrinkleResult:
    image: np.ndarray
    mask: np.ndarray
    waves: List[WrinkleWave]
    extra_mask: np.ndarray | None = None


def apply_wrinkles(
    image: np.ndarray,
    mask: np.ndarray,
    params: WrinkleParams,
    *,
    extra_mask: np.ndarray | None = None,
) -> WrinkleResult:
    """Warp the image/mask pair using wrinkle-like displacement fields."""

    if not params.enabled:
        return WrinkleResult(image=image, mask=mask, waves=[], extra_mask=extra_mask)

    height, width = mask.shape
    short_side = max(1, min(height, width))

    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    disp_x = np.zeros((height, width), dtype=np.float32)
    disp_y = np.zeros((height, width), dtype=np.float32)

    count_min, count_max = params.count_range
    if count_min > count_max:
        count_min, count_max = count_max, count_min
    wrinkle_count = random.randint(max(1, count_min), max(1, count_max))

    amp_min, amp_max = _ensure_range(params.amplitude_range, 0.0)
    sigma_min, sigma_max = _ensure_range(params.sigma_range, 1e-3)
    wave_min, wave_max = _ensure_range(params.wavelength_range, 1e-3)

    waves: List[WrinkleWave] = []

    for _ in range(wrinkle_count):
        angle = random.uniform(0.0, math.pi)
        tangent = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        cx = random.uniform(-0.25 * width, 1.25 * width)
        cy = random.uniform(-0.25 * height, 1.25 * height)
        base = np.array([cx, cy], dtype=np.float32)

        amplitude = random.uniform(amp_min, amp_max) * short_side
        sigma = random.uniform(sigma_min, sigma_max) * short_side
        wavelength = max(1.0, random.uniform(wave_min, wave_max) * short_side)

        wave = WrinkleWave(
            tangent=tangent,
            normal=normal,
            base=base,
            amplitude=float(amplitude),
            sigma=float(sigma),
            wavelength=float(wavelength),
        )
        waves.append(wave)

        dx = (xx - base[0])
        dy = (yy - base[1])
        distance = dx * normal[0] + dy * normal[1]
        parallel = dx * tangent[0] + dy * tangent[1]

        envelope = np.exp(-(distance ** 2) / (2.0 * (sigma ** 2) + 1e-6))
        phase = 2.0 * math.pi * parallel / wavelength
        displacement = amplitude * envelope * np.sin(phase)

        disp_x += displacement * normal[0]
        disp_y += displacement * normal[1]

    map_x = (xx + disp_x).astype(np.float32)
    map_y = (yy + disp_y).astype(np.float32)

    np.clip(map_x, 0, width - 1, out=map_x)
    np.clip(map_y, 0, height - 1, out=map_y)

    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    # Soft warp for mask to preserve connectivity
    warped_mask_soft = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    _, warped_mask_bin = cv2.threshold(warped_mask_soft, 50, 255, cv2.THRESH_BINARY)
    warped_mask = zhang_suen_thinning(warped_mask_bin.astype(np.uint8))
    warped_extra = None
    if extra_mask is not None:
        warped_extra = cv2.remap(
            extra_mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return WrinkleResult(image=warped_image, mask=warped_mask, waves=waves, extra_mask=warped_extra)
