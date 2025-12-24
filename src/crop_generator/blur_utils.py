from __future__ import annotations

import random

import cv2
import numpy as np

from .config import BlurParams


def _choose_kernel_size(range_tuple: tuple[int, int]) -> int:
    low, high = range_tuple
    if low > high:
        low, high = high, low
    low = max(1, low)
    high = max(low, high)
    size = random.randint(low, high)
    if size % 2 == 0:
        size += 1
    return size


def _choose_sigma(range_tuple: tuple[float, float]) -> float:
    low, high = range_tuple
    if low > high:
        low, high = high, low
    low = max(0.0, low)
    high = max(low + 1e-6, high)
    return random.uniform(low, high)


def apply_blur(image: np.ndarray, params: BlurParams) -> np.ndarray:
    if not params.enabled:
        return image

    kernel = _choose_kernel_size(params.kernel_range)
    sigma = _choose_sigma(params.sigma_range)
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
