#!/usr/bin/env python3
"""
Generate a 12‑lead ECG page that matches a classic 3×4 + long‑lead layout
on top of the supplied ECG paper grid PNG (e.g., 2200×1700). The script
tries to reproduce key visual elements from the given sample page:

- 25 mm/s paper speed (1 big box = 0.2 s)
- 10 mm/mV gain (1 mV = 2 big boxes vertically)
- 3×4 short leads (~2.5 s each, i.e., 12.5 big boxes wide)
- Bottom long rhythm strip (10 s = 50 big boxes), prefer lead II
- Lead labels inside each cell
- 1 mV calibration pulse at the left of each row (including the rhythm row)
- Footer text: "25mm/s    10mm/mV"

Input CSV must have headers among: I, II, III, aVR, aVL, aVF, V1..V6.
Missing augmented limb leads are computed from I/II when possible.

Rendering uses PIL to draw directly on the grid image without any scaling so
the red grid stays pixel‑crisp.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


LEADS_STD: Sequence[str] = ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')
SEC_PER_BOX = 0.2   # 25 mm/s
MV_PER_BOX  = 0.5   # 10 mm/mV


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


@dataclass(slots=True)
class RenderOptions:
    """High-level controls for ECG page layout."""

    label_dx_boxes: float = 0.35
    label_dy_boxes: float = 0.9
    cal_dx_boxes: float = 1.0
    conn_w_px: int = 6
    conn_h_px: int = 55
    conn_dx_boxes: float = 0.0
    unify_row_edges: bool = True
    gutter_boxes: float | None = None
    edge_margin_boxes: float | None = None
    first_lead_box: float | None = 2.0
    row_top_boxes: tuple[float, ...] | None = (18.0, 25.0, 32.0, 39.0)
    row_gap_boxes: float = 0.0
    baseline_offset_boxes: float = 0.0
    baseline_jitter_boxes: float = 0.25
    start_mode: str = 'first_nonzero'
    start_sec: float | None = None
    label_font_size: int = 36
    small_font_size: int = 16
    nonzero_epsilon: float = 1e-4


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


def detect_grid(grid_source: str | os.PathLike[str] | Image.Image) -> GridSpec:
    """Open grid image and detect big-box spacing and inner bounds."""
    if isinstance(grid_source, Image.Image):
        im = grid_source.convert("RGB")
    else:
        im = Image.open(grid_source).convert("RGB")
    width, height = im.size
    arr = np.array(im)
    # robust red grid detection
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)
    cols_red = red.sum(axis=0)
    rows_red = red.sum(axis=1)
    v_idx = np.where(cols_red > height * 0.8)[0]
    h_idx = np.where(rows_red > width * 0.8)[0]
    v_lines = np.array(_collapse_indices(v_idx), dtype=int)
    h_lines = np.array(_collapse_indices(h_idx), dtype=int)
    if len(v_lines) < 2 or len(h_lines) < 2:
        raise RuntimeError('grid detection failed')
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


def estimate_fs(y: np.ndarray) -> float:
    """Rough sampling rate guess (snap to 1000/500/360/250)."""
    rng = rng or random.Random()
    try:
        y = y[np.isfinite(y)]
        if len(y) < 1500:
            return 1000.0
        win = max(50, len(y)//200)
        yhp = y - np.convolve(y, np.ones(win)/win, 'same')
        th = np.nanpercentile(yhp, 99)
        peaks = [i for i in range(1, len(yhp)-1)
                 if yhp[i] > yhp[i-1] and yhp[i] > yhp[i+1] and yhp[i] > th]
        if len(peaks) > 5:
            rr = np.diff(peaks).astype(float)
            m = float(np.mean(rr))
            for cand in (1000, 500, 360, 250):
                if abs(m - cand) < max(60, 0.1*cand):
                    return float(cand)
    except Exception:
        pass
    return 1000.0


def _polyline(draw: ImageDraw.ImageDraw, x: np.ndarray, y: np.ndarray, color=(0,0,0), width=1):
    """Draw polyline skipping NaNs; coordinates are rounded to integer pixels."""
    assert len(x) == len(y)
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    valid = np.isfinite(x) & np.isfinite(y)
    start = None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start > 1:
                draw.line(list(zip(xi[start:i], yi[start:i])), fill=color, width=width)
            start = None
    if start is not None and len(x) - start > 1:
        draw.line(list(zip(xi[start:], yi[start:])), fill=color, width=width)


def _find_best_window(mask: np.ndarray, win: int) -> int:
    """Start index of a length‑win window with most finite samples."""
    if win <= 0 or len(mask) < win:
        return 0
    csum = np.concatenate([[0], np.cumsum(mask.astype(np.int32))])
    step = max(1, win // 10)
    best, best_cnt = 0, -1
    for s in range(0, len(mask) - win + 1, step):
        cnt = int(csum[s+win] - csum[s])
        if cnt > best_cnt:
            best, best_cnt = s, cnt
    return best


def _first_nonzero_index(y: np.ndarray, eps: float) -> int:
    """Return first index where |y| > eps; fallback to first finite sample."""
    finite = np.isfinite(y)
    nz = finite & (np.abs(y) > eps)
    if np.any(nz):
        return int(np.argmax(nz))
    if np.any(finite):
        return int(np.argmax(finite))
    return 0


def _derive_augmented_leads(leads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Fill in aVR/aVL/aVF if missing and I+II are available."""
    if 'I' in leads and 'II' in leads:
        i, ii = leads['I'], leads['II']
        n = min(len(i), len(ii))
        if 'aVR' not in leads:
            leads['aVR'] = -(i[:n] + ii[:n]) / 2.0
        if 'aVL' not in leads:
            leads['aVL'] = i[:n] - ii[:n] / 2.0
        if 'aVF' not in leads:
            leads['aVF'] = ii[:n] - i[:n] / 2.0
    return leads


def load_leads_from_csv(csv_path: str | os.PathLike[str], *, required: Iterable[str] = LEADS_STD) -> Dict[str, np.ndarray]:
    """Parse lead signals from CSV; coerces to float arrays and drops empty leads."""
    df = pd.read_csv(csv_path)
    leads: Dict[str, np.ndarray] = {}
    for ld in required:
        if ld in df.columns:
            y = pd.to_numeric(df[ld], errors='coerce').to_numpy()
            if np.isfinite(y).sum() > 10:
                leads[ld] = y
    leads = _derive_augmented_leads(leads)
    if not leads:
        raise RuntimeError('no lead data in CSV')
    return leads


def infer_sampling_rate(leads: Mapping[str, np.ndarray]) -> float:
    """Preferentially use lead II/V2/V5/I when estimating fs."""
    pick = None
    for cand in ('II', 'V2', 'V5', 'I'):
        if cand in leads:
            pick = cand
            break
    pick = pick or next(iter(leads.keys()))
    return estimate_fs(leads[pick])


def render_ecg_page(
    leads: Mapping[str, np.ndarray],
    sampling_rate: float,
    grid: GridSpec,
    *,
    options: RenderOptions,
    header_text: str | None = None,
    rng: random.Random | None = None,
) -> Image.Image:
    """Render an ECG page using preloaded lead signals and detected grid spec."""
    box_x = grid.box_x
    box_y = grid.box_y
    use_w_boxes = grid.total_boxes_x
    use_h_boxes = grid.total_boxes_y

    # Layout in boxes (match sample: 3 rows × 4 cols + bottom long rhythm)
    rows_short = 3
    rows_total = 4  # +1 rhythm row
    cols = 4
    row_h_boxes = use_h_boxes / rows_total
    col_w_boxes = use_w_boxes / cols
    short_w_boxes = 12.5  # standard 2.5 s
    # default gutters: match connector width if not specified
    gutter_boxes = options.gutter_boxes
    if gutter_boxes is None:
        default_gap = max(0.05, float(options.conn_w_px) / max(1, box_x))
        gutter_boxes = default_gap
    edge_margin_boxes = options.edge_margin_boxes
    if options.first_lead_box is not None:
        # when user pins the first column to an absolute big-box index, default to zero gap unless overridden
        if options.gutter_boxes is None:
            gutter_boxes = 0.0
        total_needed = 4 * short_w_boxes + 3 * gutter_boxes
        if total_needed > use_w_boxes:
            raise ValueError("Lead layout wider than detected grid; reduce short_w_boxes or gutters")
        max_start = use_w_boxes - total_needed
        edge_margin_boxes = min(max(0.0, options.first_lead_box), max_start)
    elif edge_margin_boxes is None:
        edge_margin_boxes = max(0.4, (use_w_boxes - 4 * short_w_boxes - 3 * gutter_boxes) / 2.0)
    # If不足，缩短短导联到刚好适配
    if edge_margin_boxes < 0:
        short_w_boxes = max(8.0, (use_w_boxes - 3 * gutter_boxes) / 4.0)
        edge_margin_boxes = 0.4
    short_sec = short_w_boxes * SEC_PER_BOX
    # margins used by short rows (统一)
    edge_margin_b = edge_margin_boxes
    # rhythm width: either standard 50 boxes centered, or align edges to short rows
    if options.unify_row_edges:
        long_left_b = edge_margin_b
        long_w_boxes = use_w_boxes - 2 * edge_margin_b
    else:
        long_w_boxes = min(50.0, use_w_boxes - 1.0)
        long_left_b = (use_w_boxes - long_w_boxes) / 2
    long_sec = long_w_boxes * SEC_PER_BOX

    def _compute_row_layout() -> tuple[list[float], list[float]]:
        tops: list[float] = []
        heights: list[float] = []
        default_height = row_h_boxes * 0.7
        row_gap = max(0.0, options.row_gap_boxes)
        if options.row_top_boxes:
            if len(options.row_top_boxes) < rows_total:
                raise ValueError("row_top_boxes must provide at least 4 values")
            prev_top = -float('inf')
            for idx in range(rows_total):
                top = float(options.row_top_boxes[idx])
                if not (0 <= top <= use_h_boxes):
                    raise ValueError(f"row_top_boxes[{idx}]={top} outside grid (0,{use_h_boxes})")
                if top <= prev_top:
                    raise ValueError("row_top_boxes must be strictly increasing")
                if idx < rows_total - 1:
                    gap = float(options.row_top_boxes[idx + 1]) - top
                else:
                    gap = use_h_boxes - top
                avail = max(1.0, gap - row_gap)
                heights.append(min(default_height, avail))
                tops.append(top)
                prev_top = top
        else:
            for idx in range(rows_total):
                cell_top = idx * row_h_boxes
                tops.append(cell_top + 0.15 * row_h_boxes)
                heights.append(default_height)
        return tops, heights

    row_tops, row_heights = _compute_row_layout()

    def _compute_row_baselines(rng_obj: random.Random) -> list[float]:
        baselines: list[float] = []
        for idx in range(rows_total):
            jitter = options.baseline_jitter_boxes
            jitter_val = rng_obj.uniform(-jitter, jitter) if jitter > 0 else 0.0
            baseline = row_tops[idx] + options.baseline_offset_boxes + jitter_val
            # keep baseline around the red row guide (±0.1 big boxes) but clamp to the row bounds
            min_allowed = max(row_tops[idx] - 0.25, 0.0)
            max_allowed = min(row_tops[idx] + 0.25, row_tops[idx] + row_heights[idx] - 0.05)
            baseline = min(max(baseline, min_allowed), max_allowed)
            baselines.append(baseline)
        return baselines

    row_baselines = _compute_row_baselines(rng)

    page = grid.image.copy()
    draw = ImageDraw.Draw(page)
    try:
        # Main label font (lead names) and small UI font
        font = ImageFont.truetype('DejaVuSans.ttf', int(options.label_font_size))
        font_small = ImageFont.truetype('DejaVuSans.ttf', int(options.small_font_size))
    except Exception:
        font = font_small = ImageFont.load_default()

    # Helper to draw a lead into its cell
    def draw_lead(y: np.ndarray, r: int, c: int, left_b: float, sec: float, label: str):
        top_b = row_tops[r]
        height_b = row_heights[r]
        base_b = row_baselines[r]

        # pixel rectangle
        lx = grid.x_from_boxes(left_b)
        rx = grid.x_from_boxes(left_b + short_w_boxes)
        ty = grid.y_from_boxes(top_b)
        by = grid.y_from_boxes(top_b + height_b)
        base_y = grid.y_from_boxes(base_b)
        cell_w = max(1, rx - lx)
        cell_h = max(1, by - ty)

        # Window selection per lead
        N = int(min(len(y), sec * sampling_rate))
        # start index selection based on mode
        if options.start_mode == 'best':
            mask = np.isfinite(y)
            s = _find_best_window(mask, N)
        elif options.start_mode == 'sequential':
            # take from the first finite sample; typical for sequentially acquired leads
            mask = np.isfinite(y)
            s = int(np.argmax(mask)) if mask.any() else 0
        elif options.start_mode == 'first_nonzero':
            s = _first_nonzero_index(y, options.nonzero_epsilon)
        else:  # 'start'
            if options.start_sec is not None and options.start_sec >= 0:
                s = int(options.start_sec * sampling_rate)
            else:
                s = 0
        seg = y[s:s+N]
        if len(seg) < 2:
            return
        x = np.linspace(lx, rx, num=len(seg), endpoint=False)
        ypx = base_y - (seg / MV_PER_BOX) * box_y
        _polyline(draw, x, ypx, color=(0,0,0), width=1)
        # label: align just below the waveform baseline near left margin
        labx = lx + int(round((options.label_dx_boxes / short_w_boxes) * cell_w))
        laby = int(round(base_y + (options.label_dy_boxes / height_b) * cell_h))
        draw.text((labx, laby), label, fill=(0,0,0), font=font)
        # small black connector block between leads: only for columns > 0 in top 3 rows
        if r < rows_short and c > 0 and options.conn_w_px > 0 and options.conn_h_px > 0:
            cx0 = lx + int(round((options.conn_dx_boxes / short_w_boxes) * cell_w))
            base = base_y
            cy0 = int(base - options.conn_h_px/2)
            cx1 = cx0 + int(options.conn_w_px)
            cy1 = cy0 + int(options.conn_h_px)
            draw.rectangle([cx0, cy0, cx1, cy1], fill=(0,0,0))

    # 1 mV calibration pulse for a row (left side of row content)
    def draw_calibration(row_idx: int):
        baseline = row_baselines[row_idx]
        base_b = baseline
        peak_b = baseline - 2.0
        base = grid.y_from_boxes(base_b)
        lx = grid.x_from_boxes(options.cal_dx_boxes)
        rx = grid.x_from_boxes(options.cal_dx_boxes + 1.0)
        step_top = grid.y_from_boxes(peak_b)
        # draw step: up, right, down
        draw.line([(lx, base), (lx, step_top)], fill=(0,0,0), width=1)
        draw.line([(lx, step_top), (rx, step_top)], fill=(0,0,0), width=1)
        draw.line([(rx, step_top), (rx, base)], fill=(0,0,0), width=1)

    # Draw 12 short leads in 3 rows × 4 cols to match reference page
    layout = [
        ['I',   'aVR', 'V1', 'V4'],
        ['II',  'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
    ]
    # 计算四段的左边界（同一行）
    lefts = [
        edge_margin_b,
        edge_margin_b + short_w_boxes + gutter_boxes,
        edge_margin_b + 2*short_w_boxes + 2*gutter_boxes,
        edge_margin_b + 3*short_w_boxes + 3*gutter_boxes,
    ]
    for r in range(rows_short):
        draw_calibration(r)
        for c, ld in enumerate(layout[r]):
            if ld in leads:
                draw_lead(leads[ld], r, c, lefts[c], short_sec, ld)
            else:
                cell_top_b  = row_tops[r]
                lx = grid.x_from_boxes(lefts[c] + 0.7)
                ty = grid.y_from_boxes(cell_top_b + 0.9 * row_heights[r])
                draw.text((lx, ty), f"{ld} (no data)", fill=(80,80,80), font=font_small)

    # Long rhythm strip at bottom row
    long_ld = 'II' if 'II' in leads else ('I' if 'I' in leads else None)
    if long_ld:
        # draw calibration for bottom row (row index 3)
        draw_calibration(3)
        left_b = long_left_b
        top_b  = row_tops[3]
        height_b = row_heights[3]
        lx = grid.x_from_boxes(left_b)
        rx = grid.x_from_boxes(left_b + long_w_boxes)
        ty = grid.y_from_boxes(top_b)
        by = grid.y_from_boxes(top_b + height_b)
        cell_w = max(1, rx - lx)
        cell_h = max(1, by - ty)
        y = leads[long_ld]
        # Window selection for long rhythm
        N = int(min(len(y), long_sec * sampling_rate))
        if options.start_mode == 'best':
            mask = np.isfinite(y)
            s = _find_best_window(mask, N)
        elif options.start_mode == 'sequential':
            mask = np.isfinite(y)
            s = int(np.argmax(mask)) if mask.any() else 0
        elif options.start_mode == 'first_nonzero':
            s = _first_nonzero_index(y, options.nonzero_epsilon)
        else:
            if options.start_sec is not None and options.start_sec >= 0:
                s = int(options.start_sec * sampling_rate)
            else:
                s = 0
        seg = y[s:s+N]
        if len(seg) < 2:
            return
        x = np.linspace(lx, rx, num=len(seg), endpoint=False)
        base = grid.y_from_boxes(row_baselines[3])
        ypx = base - (seg / MV_PER_BOX) * box_y
        _polyline(draw, x, ypx, color=(0,0,0), width=1)
        label_dx = lx + int(round(0.6 * cell_w / long_w_boxes))
        label_dy = ty + int(round(0.8 * cell_h / height_b))
        draw.text((label_dx, label_dy), f"{long_ld}", fill=(0,0,0), font=font)
    # Footer scale text centered near bottom inside grid
    footer = "25mm/s    10mm/mV"
    try:
        # PIL < 10
        tw, th = font_small.getsize(footer)
    except Exception:
        # PIL >= 10
        bbox = draw.textbbox((0,0), footer, font=font_small)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    cx = grid.x_from_boxes(use_w_boxes / 2)
    footer_boxes = 4.95 * row_h_boxes
    clamped_footer = min(footer_boxes, use_h_boxes)
    extra_boxes = max(0.0, footer_boxes - use_h_boxes)
    by = grid.y_from_boxes(clamped_footer) + int(round(extra_boxes * box_y))
    draw.text((cx - tw//2, by - th), footer, fill=(0,0,0), font=font_small)

    # Optional header (small)
    header = header_text or ""
    if header:
        draw.text((grid.x_from_boxes(1.0), grid.y0 - int(0.6*box_y)), header, fill=(0,0,0), font=font_small)

    return page


def render_ecg_page_from_csv(
    csv_path: str | os.PathLike[str],
    grid_path: str | os.PathLike[str] | Image.Image,
    *,
    options: RenderOptions | None = None,
    header_text: str | None = None,
    rng_seed: int | None = None,
    fs_override: float | None = None,
) -> Image.Image:
    """High-level helper that loads CSV + grid, then renders an ECG page."""
    options = options or RenderOptions()
    leads = load_leads_from_csv(csv_path)
    fs = fs_override if fs_override and fs_override > 0 else infer_sampling_rate(leads)
    grid = detect_grid(grid_path)
    header = header_text or os.path.basename(str(csv_path))
    rng = random.Random(rng_seed) if rng_seed is not None else None
    return render_ecg_page(leads, fs, grid, options=options, header_text=header, rng=rng)


def render_page(
    csv_path: str,
    grid_path: str,
    out_path: str,
    label_dx_boxes: float = 0.35,
    label_dy_boxes: float = 0.9,
    cal_dx_boxes: float = 1.0,
    conn_w_px: int = 6,
    conn_h_px: int = 55,
    conn_dx_boxes: float = 0.0,
    unify_row_edges: bool = True,
    gutter_boxes: float | None = None,
    edge_margin_boxes: float | None = None,
    first_lead_box: float | None = 2.0,
    row_top_boxes: Sequence[float] | None = (18.0, 25.0, 32.0, 39.0),
    row_gap_boxes: float = 0.0,
    start_mode: str = 'first_nonzero',
    start_sec: float | None = None,
    label_font_size: int = 36,
    small_font_size: int = 16,
    nonzero_epsilon: float = 1e-4,
    fs_override: float | None = None,
    baseline_offset_boxes: float = 0.0,
    baseline_jitter_boxes: float = 0.25,
) -> None:
    """Backward-compatible wrapper; prefer render_ecg_page_from_csv."""
    opts = RenderOptions(
        label_dx_boxes=label_dx_boxes,
        label_dy_boxes=label_dy_boxes,
        cal_dx_boxes=cal_dx_boxes,
        conn_w_px=conn_w_px,
        conn_h_px=conn_h_px,
        conn_dx_boxes=conn_dx_boxes,
        unify_row_edges=unify_row_edges,
        gutter_boxes=gutter_boxes,
        edge_margin_boxes=edge_margin_boxes,
        first_lead_box=first_lead_box,
        row_top_boxes=tuple(row_top_boxes) if row_top_boxes is not None else None,
        row_gap_boxes=row_gap_boxes,
        baseline_offset_boxes=baseline_offset_boxes,
        baseline_jitter_boxes=baseline_jitter_boxes,
        start_mode=start_mode,
        start_sec=start_sec,
        label_font_size=label_font_size,
        small_font_size=small_font_size,
        nonzero_epsilon=nonzero_epsilon,
    )
    image = render_ecg_page_from_csv(csv_path, grid_path, options=opts, fs_override=fs_override)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path)


def main():
    ap = argparse.ArgumentParser(description='Render ECG page with calibration and labels on supplied grid')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--header', default=None, help='Optional header text; defaults to CSV basename')
    ap.add_argument('--label-dx', type=float, default=0.35, help='Lead label x offset in big boxes from left of lead')
    ap.add_argument('--label-dy', type=float, default=0.9, help='Lead label y offset in big boxes below baseline')
    ap.add_argument('--cal-dx', type=float, default=1.0, help='Calibration step x offset in big boxes from left grid bound')
    ap.add_argument('--conn-w', type=int, default=6, help='Connector block width in pixels (default 6)')
    ap.add_argument('--conn-h', type=int, default=55, help='Connector block height in pixels (default 55)')
    ap.add_argument('--conn-dx', type=float, default=0.0, help='Connector horizontal offset from lead left boundary in big boxes (default 0.0)')
    ap.add_argument('--gutter-boxes', type=float, default=None, help='Gap between adjacent short leads in big boxes; default = conn-w/box_x')
    ap.add_argument('--edge-margin-boxes', type=float, default=None, help='Left/right outer margins in big boxes; default computed to fill width')
    ap.add_argument('--first-lead-box', type=float, default=2.0, help='Absolute big-box index for the first lead column (overrides edge margin); default 2')
    ap.add_argument('--baseline-offset-boxes', type=float, default=0.0, help='Baseline offset from each row top big-box index')
    ap.add_argument('--baseline-jitter-boxes', type=float, default=0.25, help='Random jitter for baseline (± value, in big boxes)')
    ap.add_argument('--row-top-boxes', type=str, default="18,25,32,39",
                    help='Comma-separated big-box indices for the top of each row; set to empty for auto layout')
    ap.add_argument('--row-gap-boxes', type=float, default=0.0,
                    help='Gap (in big boxes) reserved between anchored rows when --row-top-boxes is used')
    ap.add_argument('--start-mode', type=str, default='first_nonzero', choices=['start','best','sequential','first_nonzero'],
                    help="Window selection: 'start'=from record start, 'best'=per-lead best window, 'sequential'=from each lead's first available sample, 'first_nonzero' (default)=from first |value|>eps sample")
    ap.add_argument('--start-sec', type=float, default=None, help='Global start offset in seconds when start-mode=start')
    ap.add_argument('--label-font-size', type=int, default=36, help='Lead label font size (px)')
    ap.add_argument('--small-font-size', type=int, default=16, help='Small text font size (px)')
    ap.add_argument('--nonzero-eps', type=float, default=1e-4, help='Threshold for detecting first non-zero sample when start-mode=first_nonzero')
    ap.add_argument('--fs-override', type=float, default=None, help='Optional sampling rate override (Hz)')
    args = ap.parse_args()
    if args.row_top_boxes:
        row_top_boxes = tuple(float(v) for v in args.row_top_boxes.split(',') if v.strip())
    else:
        row_top_boxes = None
    opts = RenderOptions(
        label_dx_boxes=args.label_dx,
        label_dy_boxes=args.label_dy,
        cal_dx_boxes=args.cal_dx,
        conn_w_px=args.conn_w,
        conn_h_px=args.conn_h,
        conn_dx_boxes=args.conn_dx,
        gutter_boxes=args.gutter_boxes,
        edge_margin_boxes=args.edge_margin_boxes,
        first_lead_box=args.first_lead_box,
        baseline_offset_boxes=args.baseline_offset_boxes,
        baseline_jitter_boxes=args.baseline_jitter_boxes,
        row_top_boxes=row_top_boxes,
        row_gap_boxes=args.row_gap_boxes,
        start_mode=args.start_mode,
        start_sec=args.start_sec,
        label_font_size=args.label_font_size,
        small_font_size=args.small_font_size,
        nonzero_epsilon=args.nonzero_eps,
    )
    rng_seed = hash(args.header or Path(args.csv).stem) & 0xFFFFFFFF
    image = render_ecg_page_from_csv(
        args.csv,
        args.grid,
        options=opts,
        header_text=args.header,
        rng_seed=rng_seed,
        fs_override=args.fs_override,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


if __name__ == '__main__':
    main()
