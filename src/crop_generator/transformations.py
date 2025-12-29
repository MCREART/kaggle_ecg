from __future__ import annotations

import random
import cv2
import numpy as np

from dataclasses import dataclass

from .config import TransformMode, TransformParams
from .mask_utils import zhang_suen_thinning


@dataclass(frozen=True)
class TransformResult:
    image: np.ndarray
    mask: np.ndarray
    matrix: np.ndarray
    mode: TransformMode


def random_affine_matrix(
    width: int, height: int, *, max_rotate: float, max_shift: float, max_scale: float
) -> np.ndarray:
    center = (width * 0.5, height * 0.5)
    angle = random.uniform(-max_rotate, max_rotate)
    scale = 1 + random.uniform(-max_scale, max_scale)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    shift_x = random.uniform(-max_shift, max_shift) * width
    shift_y = random.uniform(-max_shift, max_shift) * height
    matrix[:, 2] += (shift_x, shift_y)
    return matrix


def random_perspective_matrix(
    width: int, height: int, *, jitter: float
) -> np.ndarray:
    def jitter_point(x: float, y: float) -> np.ndarray:
        offset_x = random.uniform(-jitter, jitter) * width
        offset_y = random.uniform(-jitter, jitter) * height
        jittered_x = float(np.clip(x + offset_x, 0.0, width - 1.0))
        jittered_y = float(np.clip(y + offset_y, 0.0, height - 1.0))
        return np.array([jittered_x, jittered_y], dtype=np.float32)

    src = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    dst = np.array(
        [
            jitter_point(0, 0),
            jitter_point(width - 1, 0),
            jitter_point(width - 1, height - 1),
            jitter_point(0, height - 1),
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def apply_transform(
    image: np.ndarray,
    mask: np.ndarray,
    params: TransformParams,
) -> TransformResult:
    """Apply the configured transform to image and mask."""

    height, width = mask.shape
    mode = params.mode
    if mode == "none":
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return TransformResult(image=image, mask=mask, matrix=identity, mode=mode)

    if mode == "affine":
        matrix = random_affine_matrix(
            width,
            height,
            max_rotate=params.max_rotate,
            max_shift=params.max_shift,
            max_scale=params.max_scale,
        )
        warped_img = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        # Soft warp for mask to preserve connectivity
        warped_mask_soft = cv2.warpAffine(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        _, warped_mask_bin = cv2.threshold(warped_mask_soft, 50, 255, cv2.THRESH_BINARY)
        # Re-thin to ensuring 1px
        warped_mask = zhang_suen_thinning(warped_mask_bin.astype(np.uint8))
        return TransformResult(image=warped_img, mask=warped_mask, matrix=matrix, mode=mode)

    if mode == "perspective":
        matrix = random_perspective_matrix(
            width,
            height,
            jitter=params.perspective_jitter,
        )
        warped_img = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        # Soft warp for mask
        warped_mask_soft = cv2.warpPerspective(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        _, warped_mask_bin = cv2.threshold(warped_mask_soft, 50, 255, cv2.THRESH_BINARY)
        # Re-thin
        warped_mask = zhang_suen_thinning(warped_mask_bin.astype(np.uint8))
        return TransformResult(image=warped_img, mask=warped_mask, matrix=matrix, mode=mode)

    raise ValueError(f"Unsupported transform mode: {mode}")
