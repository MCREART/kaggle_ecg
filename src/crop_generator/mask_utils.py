from __future__ import annotations

import numpy as np


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Perform Zhang-Suen thinning to obtain a 1-pixel skeleton."""

    img = binary.copy()
    changing = True
    rows, cols = img.shape

    while changing:
        changing = False
        to_remove: list[tuple[int, int]] = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] == 0:
                    continue
                neighborhood = img[i - 1 : i + 2, j - 1 : j + 2]
                neighbors = [
                    neighborhood[0, 1],
                    neighborhood[0, 2],
                    neighborhood[1, 2],
                    neighborhood[2, 2],
                    neighborhood[2, 1],
                    neighborhood[2, 0],
                    neighborhood[1, 0],
                    neighborhood[0, 0],
                ]
                transitions = sum(
                    neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1 for k in range(8)
                )
                neighbor_count = sum(neighbors)
                if (
                    2 <= neighbor_count <= 6
                    and transitions == 1
                    and neighbors[0] * neighbors[2] * neighbors[4] == 0
                    and neighbors[2] * neighbors[4] * neighbors[6] == 0
                ):
                    to_remove.append((i, j))
        if to_remove:
            changing = True
            for i, j in to_remove:
                img[i, j] = 0

        to_remove = []
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] == 0:
                    continue
                neighborhood = img[i - 1 : i + 2, j - 1 : j + 2]
                neighbors = [
                    neighborhood[0, 1],
                    neighborhood[0, 2],
                    neighborhood[1, 2],
                    neighborhood[2, 2],
                    neighborhood[2, 1],
                    neighborhood[2, 0],
                    neighborhood[1, 0],
                    neighborhood[0, 0],
                ]
                transitions = sum(
                    neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1 for k in range(8)
                )
                neighbor_count = sum(neighbors)
                if (
                    2 <= neighbor_count <= 6
                    and transitions == 1
                    and neighbors[0] * neighbors[2] * neighbors[6] == 0
                    and neighbors[0] * neighbors[4] * neighbors[6] == 0
                ):
                    to_remove.append((i, j))
        if to_remove:
            changing = True
            for i, j in to_remove:
                img[i, j] = 0

    return img


def resize_skeleton(binary: np.ndarray, out_size: int) -> np.ndarray:
    """
    Resize a skeletonized mask by reprojecting white pixels to the new grid.

    This method avoids interpolation artifacts and keeps single-pixel lines.
    """

    coords = np.column_stack(np.nonzero(binary))
    if coords.size == 0:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    height, width = binary.shape
    scale_y = out_size / float(height)
    scale_x = out_size / float(width)
    scaled_rows = np.rint(coords[:, 0] * scale_y).astype(int)
    scaled_cols = np.rint(coords[:, 1] * scale_x).astype(int)
    scaled_rows = np.clip(scaled_rows, 0, out_size - 1)
    scaled_cols = np.clip(scaled_cols, 0, out_size - 1)

    out = np.zeros((out_size, out_size), dtype=np.uint8)
    out[scaled_rows, scaled_cols] = 255
    return out
