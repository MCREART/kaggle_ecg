from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(slots=True)
class GridSpec:
    """Detected grid geometry with per-interval spacing."""

    image: Image.Image
    x_lines: np.ndarray
    y_lines: np.ndarray
    x_offsets: np.ndarray
    y_offsets: np.ndarray
    x_intervals: np.ndarray
    y_intervals: np.ndarray
    x0: int
    x1: int
    y0: int
    y1: int
    box_x: float
    box_y: float

    @property
    def inner_width(self) -> int:
        return int(self.x_offsets[-1])

    @property
    def inner_height(self) -> int:
        return int(self.y_offsets[-1])

    @property
    def total_boxes_x(self) -> int:
        return len(self.x_intervals)

    @property
    def total_boxes_y(self) -> int:
        return len(self.y_intervals)

    def _boxes_to_offset(self, boxes: float, offsets: np.ndarray, intervals: np.ndarray) -> float:
        total = len(intervals)
        if total == 0:
            return 0.0
        if boxes <= 0:
            return 0.0
        if boxes >= total:
            return float(offsets[-1])
        i = int(boxes)
        frac = boxes - i
        base = offsets[i]
        width = intervals[i]
        return float(base + frac * width)

    def x_from_boxes(self, boxes: float) -> int:
        return int(round(self.x0 + self._boxes_to_offset(boxes, self.x_offsets, self.x_intervals)))

    def y_from_boxes(self, boxes: float) -> int:
        return int(round(self.y0 + self._boxes_to_offset(boxes, self.y_offsets, self.y_intervals)))


def _collapse_indices(idx: np.ndarray) -> list[int]:
    out: list[int] = []
    if len(idx) == 0:
        return out
    s = p = int(idx[0])
    for c in map(int, idx[1:]):
        if c == p + 1:
            p = c
        else:
            out.append((s + p) // 2)
            s = p = c
    out.append((s + p) // 2)
    return out


def detect_grid(grid_source: str | os.PathLike[str] | Image.Image | np.ndarray) -> GridSpec:
    """Open grid image and detect big-box spacing and inner bounds."""
    if isinstance(grid_source, Image.Image):
        im = grid_source.convert("RGB")
    elif isinstance(grid_source, np.ndarray):
         im = Image.fromarray(grid_source).convert("RGB")
    else:
        im = Image.open(grid_source).convert("RGB")
    width, height = im.size
    arr = np.array(im)
    # robust red grid detection
    red = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100) # Relaxed color check
    cols_red = red.sum(axis=0)
    rows_red = red.sum(axis=1)
    
    # Relaxed length threshold: 50% instead of 80%
    v_idx = np.where(cols_red > height * 0.5)[0]
    h_idx = np.where(rows_red > width * 0.5)[0]
    
    v_lines = np.array(_collapse_indices(v_idx), dtype=int)
    h_lines = np.array(_collapse_indices(h_idx), dtype=int)
    if len(v_lines) < 2 or len(h_lines) < 2:
        raise RuntimeError(f'grid detection failed: localized {len(h_lines)} H-lines, {len(v_lines)} V-lines')
    x_intervals = np.diff(v_lines)
    y_intervals = np.diff(h_lines)
    x_offsets = np.concatenate(([0], np.cumsum(x_intervals)))
    y_offsets = np.concatenate(([0], np.cumsum(y_intervals)))
    box_x = float(np.median(x_intervals))
    box_y = float(np.median(y_intervals))
    x0, x1 = int(v_lines[0]), int(v_lines[-1])
    y0, y1 = int(h_lines[0]), int(h_lines[-1])
    return GridSpec(
        image=im,
        x_lines=v_lines,
        y_lines=h_lines,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        x_intervals=x_intervals,
        y_intervals=y_intervals,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        box_x=box_x,
        box_y=box_y,
    )
