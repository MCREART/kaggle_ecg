from __future__ import annotations

import cv2
import numpy as np

from .config import GrayParams


def apply_grayscale(image: np.ndarray, params: GrayParams) -> np.ndarray:
    if not params.enabled:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if params.preserve_channels:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray
