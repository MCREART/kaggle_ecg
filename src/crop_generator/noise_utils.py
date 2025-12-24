from __future__ import annotations

import random

import numpy as np

from .config import NoiseParams


def _choose_sigma(range_tuple: tuple[float, float]) -> float:
    low, high = range_tuple
    if low > high:
        low, high = high, low
    low = max(0.0, low)
    high = max(low + 1e-6, high)
    return random.uniform(low, high)


def apply_noise(image: np.ndarray, params: NoiseParams) -> np.ndarray:
    if not params.enabled:
        return image

    sigma = _choose_sigma(params.sigma_range)
    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)

    if image.dtype != np.float32:
        float_image = image.astype(np.float32)
    else:
        float_image = image.copy()

    noisy = float_image + noise
    np.clip(noisy, 0, 255, out=noisy)
    return noisy.astype(image.dtype)
