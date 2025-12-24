from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import cv2
import numpy as np

from .mask_utils import zhang_suen_thinning


_NEIGHBOR_OFFSETS = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
)


def _skeleton_to_segments(skeleton: np.ndarray) -> List[np.ndarray]:
    points = np.argwhere(skeleton > 0)
    if points.size == 0:
        return []

    occupied = {tuple(coord) for coord in points.tolist()}
    segments: List[np.ndarray] = []

    for y, x in occupied:
        has_neighbor = False
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny = y + dy
            nx = x + dx
            if (ny, nx) in occupied:
                has_neighbor = True
                start = np.array([x, y], dtype=np.float32)
                end = np.array([nx, ny], dtype=np.float32)
                segments.append(np.stack([start, end], axis=0))
        if not has_neighbor:
            point = np.array([x, y], dtype=np.float32)
            segments.append(np.stack([point, point], axis=0))
    return segments


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = hom @ matrix.T
    return transformed.astype(np.float32)


def _apply_perspective(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = hom @ matrix.T
    w = np.clip(transformed[:, 2], 1e-6, None)
    out = transformed[:, :2] / w[:, None]
    return out.astype(np.float32)


def _clip_segment_to_rect(
    start: np.ndarray,
    end: np.ndarray,
    rect_x: float,
    rect_y: float,
    rect_w: float,
    rect_h: float,
) -> np.ndarray | None:
    x_min = rect_x
    y_min = rect_y
    x_max = rect_x + rect_w
    y_max = rect_y + rect_h

    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def _out_code(x: float, y: float) -> int:
        code = INSIDE
        if x < x_min:
            code |= LEFT
        elif x > x_max:
            code |= RIGHT
        if y < y_min:
            code |= TOP
        elif y > y_max:
            code |= BOTTOM
        return code

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])

    out0 = _out_code(x0, y0)
    out1 = _out_code(x1, y1)

    while True:
        if not (out0 | out1):
            return np.array([[x0, y0], [x1, y1]], dtype=np.float32)
        if out0 & out1:
            return None
        out_code = out0 or out1
        if out_code & TOP:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (y_min - y0) / (y1 - y0)
            y = y_min
        elif out_code & BOTTOM:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (y_max - y0) / (y1 - y0)
            y = y_max
        elif out_code & RIGHT:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (x_max - x0) / (x1 - x0)
            x = x_max
        else:  # LEFT
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (x_min - x0) / (x1 - x0)
            x = x_min

        if out_code == out0:
            x0, y0 = x, y
            out0 = _out_code(x0, y0)
        else:
            x1, y1 = x, y
            out1 = _out_code(x1, y1)


@dataclass(frozen=True)
class WrinkleWave:
    tangent: np.ndarray
    normal: np.ndarray
    base: np.ndarray
    amplitude: float
    sigma: float
    wavelength: float


class VectorMask:
    def __init__(self, segments: Sequence[np.ndarray], width: int, height: int) -> None:
        self.segments = [segment.astype(np.float32, copy=False) for segment in segments]
        self.width = int(width)
        self.height = int(height)

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "VectorMask":
        binary = (mask > 0).astype(np.uint8)
        skeleton = zhang_suen_thinning(binary)
        segments = _skeleton_to_segments(skeleton)
        height, width = mask.shape
        return cls(segments, width, height)

    def warp_affine(self, matrix: np.ndarray, width: int, height: int) -> "VectorMask":
        transformed: List[np.ndarray] = []
        for segment in self.segments:
            new_segment = _apply_affine(segment, matrix)
            new_segment[:, 0] = np.clip(new_segment[:, 0], 0.0, width - 1.0)
            new_segment[:, 1] = np.clip(new_segment[:, 1], 0.0, height - 1.0)
            transformed.append(new_segment)
        return VectorMask(transformed, width, height)

    def warp_perspective(self, matrix: np.ndarray, width: int, height: int) -> "VectorMask":
        transformed: List[np.ndarray] = []
        for segment in self.segments:
            new_segment = _apply_perspective(segment, matrix)
            new_segment[:, 0] = np.clip(new_segment[:, 0], 0.0, width - 1.0)
            new_segment[:, 1] = np.clip(new_segment[:, 1], 0.0, height - 1.0)
            transformed.append(new_segment)
        return VectorMask(transformed, width, height)

    def warp_wrinkles(self, waves: Iterable[WrinkleWave], width: int, height: int) -> "VectorMask":
        waves = list(waves)
        if not waves:
            return VectorMask(self.segments, width, height)

        transformed: List[np.ndarray] = []
        for segment in self.segments:
            pts = segment.copy()
            for wave in waves:
                dx = pts[:, 0] - wave.base[0]
                dy = pts[:, 1] - wave.base[1]
                distance = dx * wave.normal[0] + dy * wave.normal[1]
                parallel = dx * wave.tangent[0] + dy * wave.tangent[1]
                denom = 2.0 * (wave.sigma ** 2) + 1e-6
                envelope = np.exp(-(distance ** 2) / denom)
                phase = 2.0 * np.pi * parallel / max(wave.wavelength, 1e-6)
                displacement = wave.amplitude * envelope * np.sin(phase)
                pts[:, 0] += displacement * wave.normal[0]
                pts[:, 1] += displacement * wave.normal[1]
            pts[:, 0] = np.clip(pts[:, 0], 0.0, width - 1.0)
            pts[:, 1] = np.clip(pts[:, 1], 0.0, height - 1.0)
            transformed.append(pts.astype(np.float32))
        return VectorMask(transformed, width, height)

    def translate(self, dx: float, dy: float, new_width: int, new_height: int) -> "VectorMask":
        offset = np.array([dx, dy], dtype=np.float32)
        translated = [segment + offset for segment in self.segments]
        return VectorMask(translated, new_width, new_height)

    def render_crop_local(
        self,
        rect_x: int,
        rect_y: int,
        rect_w: int,
        rect_h: int,
    ) -> np.ndarray:
        canvas = np.zeros((rect_h, rect_w), dtype=np.uint8)
        if rect_w <= 0 or rect_h <= 0 or not self.segments:
            return canvas

        for segment in self.segments:
            clipped = _clip_segment_to_rect(segment[0], segment[1], rect_x, rect_y, rect_w, rect_h)
            if clipped is None:
                continue
            local = clipped - np.array([rect_x, rect_y], dtype=np.float32)
            scaled = np.clip(np.rint(local), 0, [rect_w - 1, rect_h - 1]).astype(np.int32)
            p0 = (int(scaled[0, 0]), int(scaled[0, 1]))
            p1 = (int(scaled[1, 0]), int(scaled[1, 1]))
            if p0 == p1:
                canvas[p0[1], p0[0]] = 255
            else:
                cv2.line(canvas, p0, p1, 255, thickness=1, lineType=cv2.LINE_8)
        return canvas

    def render_crop(
        self,
        rect_x: int,
        rect_y: int,
        rect_w: int,
        rect_h: int,
        out_size: int,
    ) -> np.ndarray:
        local = self.render_crop_local(rect_x, rect_y, rect_w, rect_h)
        if out_size == rect_h == rect_w:
            return local
        if out_size <= 0:
            raise ValueError("out_size must be positive")
        if rect_w <= 0 or rect_h <= 0:
            return np.zeros((out_size, out_size), dtype=np.uint8)
        resized = cv2.resize(local, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        return resized
