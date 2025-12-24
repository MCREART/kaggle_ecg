#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from gen_ecg_page import (
    MV_PER_BOX,
    SEC_PER_BOX,
    RenderOptions,
    _find_best_window,
    _first_nonzero_index,
    detect_grid,
    infer_sampling_rate,
    load_leads_from_csv,
    _polyline,
)


def _compute_row_layout(use_h_boxes: float, row_h_boxes: float, options: RenderOptions):
    rows_total = 4
    default_height = row_h_boxes * 0.7
    row_gap = max(0.0, options.row_gap_boxes)
    tops: list[float] = []
    heights: list[float] = []

    if options.row_top_boxes:
        if len(options.row_top_boxes) < rows_total:
            raise ValueError("row_top_boxes must provide at least 4 entries")
        prev_top = -float("inf")
        for idx in range(rows_total):
            top = float(options.row_top_boxes[idx])
            if not (0 <= top <= use_h_boxes):
                raise ValueError(f"row_top_boxes[{idx}]={top} outside grid height {use_h_boxes}")
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


def render_wave_mask(
    csv_path: str | Path,
    grid_path: str | Path,
    out_path: str | Path,
    *,
    options: RenderOptions,
    line_width: int = 2,
    fs_override: float | None = None,
    rng_seed: int | None = None,
) -> None:
    leads = load_leads_from_csv(csv_path)
    fs = fs_override if fs_override and fs_override > 0 else infer_sampling_rate(leads)
    grid = detect_grid(grid_path)

    box_x = grid.box_x
    use_w_boxes = grid.total_boxes_x
    use_h_boxes = grid.total_boxes_y

    rows_short = 3
    rows_total = 4
    short_w_boxes = 12.5
    gutter_boxes = options.gutter_boxes
    if gutter_boxes is None:
        gutter_boxes = max(0.05, float(options.conn_w_px) / max(1, box_x))
    edge_margin_boxes = options.edge_margin_boxes
    if options.first_lead_box is not None:
        if options.gutter_boxes is None:
            gutter_boxes = 0.0
        total_needed = 4 * short_w_boxes + 3 * gutter_boxes
        if total_needed > use_w_boxes:
            raise ValueError("Lead layout wider than detected grid; reduce width/gutter")
        max_start = use_w_boxes - total_needed
        edge_margin_boxes = min(max(0.0, options.first_lead_box), max_start)
    elif edge_margin_boxes is None:
        edge_margin_boxes = max(0.4, (use_w_boxes - 4 * short_w_boxes - 3 * gutter_boxes) / 2.0)
    if edge_margin_boxes < 0:
        short_w_boxes = max(8.0, (use_w_boxes - 3 * gutter_boxes) / 4.0)
        edge_margin_boxes = 0.4
    short_sec = short_w_boxes * SEC_PER_BOX
    long_left_b = edge_margin_boxes
    long_w_boxes = use_w_boxes - 2 * edge_margin_boxes
    long_sec = long_w_boxes * SEC_PER_BOX

    row_h_boxes = use_h_boxes / rows_total
    row_tops, row_heights = _compute_row_layout(use_h_boxes, row_h_boxes, options)
    rng = random.Random(rng_seed) if rng_seed is not None else random.Random()
    row_baselines = []
    for idx in range(rows_total):
        jitter = options.baseline_jitter_boxes
        jitter_val = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
        baseline = row_tops[idx] + options.baseline_offset_boxes + jitter_val
        min_allowed = max(row_tops[idx] - 0.25, 0.0)
        max_allowed = min(row_tops[idx] + 0.25, row_tops[idx] + row_heights[idx] - 0.05)
        baseline = min(max(baseline, min_allowed), max_allowed)
        row_baselines.append(baseline)

    lefts = [
        edge_margin_boxes,
        edge_margin_boxes + short_w_boxes + gutter_boxes,
        edge_margin_boxes + 2 * short_w_boxes + 2 * gutter_boxes,
        edge_margin_boxes + 3 * short_w_boxes + 3 * gutter_boxes,
    ]

    canvas = Image.new("L", grid.image.size, color=0)
    draw = ImageDraw.Draw(canvas)

    def draw_lead(y: np.ndarray, r: int, left_b: float, sec: float):
        if y is None:
            return
        top_b = row_tops[r]
        height_b = row_heights[r]
        lx = grid.x_from_boxes(left_b)
        rx = grid.x_from_boxes(left_b + short_w_boxes)
        ty = grid.y_from_boxes(top_b)
        by = grid.y_from_boxes(top_b + height_b)
        N = int(min(len(y), sec * fs))
        if options.start_mode == "best":
            mask = np.isfinite(y)
            s = _find_best_window(mask, N)
        elif options.start_mode == "sequential":
            mask = np.isfinite(y)
            s = int(np.argmax(mask)) if mask.any() else 0
        elif options.start_mode == "first_nonzero":
            s = _first_nonzero_index(y, options.nonzero_epsilon)
        else:
            if options.start_sec is not None and options.start_sec >= 0:
                s = int(options.start_sec * fs)
            else:
                s = 0
        seg = y[s : s + N]
        if len(seg) < 2:
            return
        x = np.linspace(lx, rx, num=len(seg), endpoint=False)
        base = grid.y_from_boxes(row_baselines[r])
        ypx = base - (seg / MV_PER_BOX) * grid.box_y
        _polyline(draw, x, ypx, color=255, width=line_width)

    layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]
    for r in range(rows_short):
        for c, lead_name in enumerate(layout[r]):
            draw_lead(leads.get(lead_name), r, lefts[c], short_sec)

    long_ld = "II" if "II" in leads else ("I" if "I" in leads else None)
    if long_ld:
        top_b = row_tops[3]
        height_b = row_heights[3]
        lx = grid.x_from_boxes(long_left_b)
        rx = grid.x_from_boxes(long_left_b + long_w_boxes)
        ty = grid.y_from_boxes(top_b)
        by = grid.y_from_boxes(top_b + height_b)
        ylong = leads[long_ld]
        N = int(min(len(ylong), long_sec * fs))
        if options.start_mode == "best":
            mask = np.isfinite(ylong)
            s = _find_best_window(mask, N)
        elif options.start_mode == "sequential":
            mask = np.isfinite(ylong)
            s = int(np.argmax(mask)) if mask.any() else 0
        elif options.start_mode == "first_nonzero":
            s = _first_nonzero_index(ylong, options.nonzero_epsilon)
        else:
            if options.start_sec is not None and options.start_sec >= 0:
                s = int(options.start_sec * fs)
            else:
                s = 0
        seg = ylong[s : s + N]
        if len(seg) < 2:
            return
        x = np.linspace(lx, rx, num=len(seg), endpoint=False)
        base = grid.y_from_boxes(row_baselines[3])
        ypx = base - (seg / MV_PER_BOX) * grid.box_y
        _polyline(draw, x, ypx, color=255, width=line_width)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render waveform-only mask (white traces on black)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-lead-box", type=float, default=2.0)
    ap.add_argument("--row-top-boxes", type=str, default="18,25,32,39")
    ap.add_argument("--row-gap-boxes", type=float, default=0.0)
    ap.add_argument("--gutter-boxes", type=float, default=None)
    ap.add_argument("--edge-margin-boxes", type=float, default=None)
    ap.add_argument("--baseline-offset-boxes", type=float, default=0.0)
    ap.add_argument("--baseline-jitter-boxes", type=float, default=0.25)
    ap.add_argument("--start-mode", type=str, default="first_nonzero", choices=["start", "best", "sequential", "first_nonzero"])
    ap.add_argument("--start-sec", type=float, default=None)
    ap.add_argument("--nonzero-eps", type=float, default=1e-4)
    ap.add_argument("--line-width", type=int, default=2)
    ap.add_argument("--fs-override", type=float, default=None)
    ap.add_argument("--rng-seed", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.row_top_boxes:
        row_top_boxes: Sequence[float] | None = tuple(float(v) for v in args.row_top_boxes.split(",") if v.strip())
    else:
        row_top_boxes = None
    opts = RenderOptions(
        first_lead_box=args.first_lead_box,
        row_top_boxes=row_top_boxes,
        row_gap_boxes=args.row_gap_boxes,
        gutter_boxes=args.gutter_boxes,
        edge_margin_boxes=args.edge_margin_boxes,
        start_mode=args.start_mode,
        start_sec=args.start_sec,
        nonzero_epsilon=args.nonzero_eps,
        baseline_offset_boxes=args.baseline_offset_boxes,
        baseline_jitter_boxes=args.baseline_jitter_boxes,
    )
    render_wave_mask(
        args.csv,
        args.grid,
        args.out,
        options=opts,
        line_width=args.line_width,
        fs_override=args.fs_override,
        rng_seed=args.rng_seed,
    )


if __name__ == "__main__":
    main()
