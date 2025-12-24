from __future__ import annotations

from typing import Final

import numpy as np
import cv2
import warnings

try:  # pragma: no cover - optional dependency
    import torch

    _TORCH_AVAILABLE: Final[bool] = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch not installed or no GPU
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

_GPU_RUNTIME_ENABLED = _TORCH_AVAILABLE


def available() -> bool:
    """Return True when torch with CUDA is available."""
    return _GPU_RUNTIME_ENABLED


def _disable_gpu(reason: str) -> None:
    global _GPU_RUNTIME_ENABLED
    if _GPU_RUNTIME_ENABLED:
        warnings.warn(f"GPU 加速已禁用：{reason}", RuntimeWarning)
    _GPU_RUNTIME_ENABLED = False


def _ensure_available() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("GPU 加速不可用，请检查是否安装了 PyTorch 且启用了 CUDA")
    if not _GPU_RUNTIME_ENABLED:
        raise RuntimeError("GPU 加速已禁用，正在使用 CPU 回退逻辑")


def _to_tensor(mask: np.ndarray) -> "torch.Tensor":
    _ensure_available()
    device = torch.device("cuda")
    tensor = torch.from_numpy((mask > 0).astype(np.float32))
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def _from_tensor(tensor: "torch.Tensor") -> np.ndarray:
    array = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    return (array > 0.5).astype(np.uint8)


def binary_erode(mask: np.ndarray, kernel_size: int, iterations: int = 1) -> np.ndarray:
    """Binary erosion implemented via dilation of the inverted mask."""

    if not _GPU_RUNTIME_ENABLED:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=iterations)

    try:
        inv = 1 - binary_dilate(1 - (mask > 0).astype(np.uint8), kernel_size, iterations)
        return inv
    except RuntimeError as err:  # pragma: no cover - GPU failure path
        _disable_gpu(str(err))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=iterations)


def binary_closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Binary closing (dilation followed by erosion) on GPU."""

    if not _GPU_RUNTIME_ENABLED:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    try:
        dilated = binary_dilate(mask, kernel_size)
        return binary_erode(dilated, kernel_size)
    except RuntimeError as err:  # pragma: no cover - GPU failure path
        _disable_gpu(str(err))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel)


def cpu_binary_dilate(mask: np.ndarray, kernel_size: int, iterations: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=iterations)


def binary_dilate(mask: np.ndarray, kernel_size: int, iterations: int = 1) -> np.ndarray:
    """Binary dilation using max pooling on GPU with CPU fallback."""

    if not _GPU_RUNTIME_ENABLED:
        return cpu_binary_dilate(mask, kernel_size, iterations)

    _ensure_available()
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    iterations = max(1, int(iterations))
    try:
        tensor = _to_tensor(mask)
        pad = kernel_size // 2
        for _ in range(iterations):
            tensor = torch.nn.functional.max_pool2d(
                tensor,
                kernel_size,
                stride=1,
                padding=pad,
            )
        return _from_tensor(tensor)
    except RuntimeError as err:  # pragma: no cover - GPU failure path
        _disable_gpu(str(err))
        return cpu_binary_dilate(mask, kernel_size, iterations)
