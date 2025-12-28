#!/usr/bin/env python3
"""
批量渲染 ECG 波形页面及对应的波形掩模。

给定 CSV 根目录和纸张网格 PNG，本脚本会：
1. 调用 gen_ecg_page.render_ecg_page_from_csv 生成带网格、标签的整页图片；
2. 调用 gen_wave_mask.render_wave_mask 生成黑底白线的波形掩模；
3. 可将整页 PNG 与掩模 PNG 输出到独立目录（--image-dir / --mask-dir），默认仍写在同一目录。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import csv

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_SRC = REPO_ROOT / "ecg_generator_with_train_src"
if str(GEN_SRC) not in sys.path:
    sys.path.insert(0, str(GEN_SRC))

from gen_ecg_page import RenderOptions, render_ecg_page_from_csv  # noqa: E402
from gen_wave_mask import render_wave_mask  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def iter_csv_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.csv")):
        if path.is_file():
            yield path


def parse_row_tops(arg: str | None) -> Sequence[float] | None:
    if not arg:
        return None
    vals = [v.strip() for v in arg.split(",")]
    floats = [float(v) for v in vals if v]
    return tuple(floats) if floats else None


def build_render_options(args: argparse.Namespace) -> RenderOptions:
    return RenderOptions(
        cal_dx_boxes=args.cal_dx,
        conn_w_px=args.conn_w,
        conn_h_px=args.conn_h,
        conn_dx_boxes=args.conn_dx,
        gutter_boxes=args.gutter_boxes,
        edge_margin_boxes=args.edge_margin_boxes,
        first_lead_box=args.first_lead_box,
        row_top_boxes=parse_row_tops(args.row_top_boxes),
        row_gap_boxes=args.row_gap_boxes,
        baseline_offset_boxes=args.baseline_offset_boxes,
        baseline_jitter_boxes=args.baseline_jitter_boxes,
        start_mode=args.start_mode,
        start_sec=args.start_sec,
        nonzero_epsilon=args.nonzero_eps,
        label_font_size=args.label_font_size,
        small_font_size=args.small_font_size,
    )


def load_fs_lookup(meta_path: Path | None) -> Mapping[str, float]:
    if meta_path is None or not meta_path.exists():
        return {}
    mapping: dict[str, float] = {}
    with meta_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id")
            fs = row.get("fs")
            if not rid or not fs:
                continue
            try:
                mapping[rid] = float(fs)
            except ValueError:
                continue
    return mapping


from PIL import ImageDraw, ImageFont
import random

from PIL import ImageDraw, ImageFont
import random
import datetime

def add_synthetic_text(image, rng_seed):
    """Adds structured synthetic metadata text to the image (Header/Footer)."""
    random.seed(rng_seed)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    
    # Try multiple common Sans fonts
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "DejaVuSans.ttf" 
    ]
    
    font_path = None
    for p in font_candidates:
        try:
            # Test open
            ImageFont.truetype(p, 20)
            font_path = p
            break
        except Exception:
            continue
            
    try:
        if font_path:
            # Header fonts
            font_header = ImageFont.truetype(font_path, 40)
            font_sub = ImageFont.truetype(font_path, 30)
            font_small = ImageFont.truetype(font_path, 24)
        else:
            font_header = ImageFont.load_default()
            font_sub = ImageFont.load_default()
            font_small = ImageFont.load_default()
    except IOError:
        font_header = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # --- Header Information ---
    # Top Left: Demographics
    genders = ["Male", "Female", "M", "F"]
    pid = "".join([str(random.randint(0, 9)) for _ in range(8)])
    age = random.randint(18, 99)
    gender = random.choice(genders)
    
    # Random realistic date
    year = random.randint(2018, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    date_str = f"{day:02d}-{month:02d}-{year} {hour:02d}:{minute:02d}"
    
    # Name (Initials or random string)
    name = f"Patient_{random.randint(1000,9999)}"
    
    # Draw Top Left Block
    # Line 1: Name ID
    draw.text((40, 30), f"Name: {name}", fill="black", font=font_header)
    draw.text((40, 80), f"ID: {pid}", fill="black", font=font_sub)
    # Line 2: Date Age Sex
    draw.text((40, 120), f"Date: {date_str}   Age: {age}yr   Sex: {gender}", fill="black", font=font_sub)

    # Top Right: Hospital / Institution
    hospitals = [
        "General Hospital", "Memorial Medical Center", "University Hospital", 
        "Cardiology Dept", "St. John's", "City Heart Center"
    ]
    hosp_name = random.choice(hospitals)
    bbox = draw.textbbox((0, 0), hosp_name, font=font_header)
    tw = bbox[2] - bbox[0]
    draw.text((w - tw - 50, 30), hosp_name, fill="black", font=font_header)
    
    # --- Footer Information ---
    # Standard technicals
    speed = "25 mm/s"
    gain = "10 mm/mV"
    filters = ["150Hz", "100Hz", "40Hz", "Notch On"]
    filt = random.choice(filters)
    
    tech_str = f"{speed}   {gain}   {filt}"
    
    # Bottom Center or Right
    bbox = draw.textbbox((0, 0), tech_str, font=font_sub)
    tw = bbox[2] - bbox[0]
    # Place at bottom margin
    draw.text((w // 2 - tw // 2, h - 60), tech_str, fill="black", font=font_sub)

    return image


def render_single(
    csv_path: Path,
    grid_path: Path,
    image_dir: Path,
    mask_dir: Path,
    options: RenderOptions,
    *,
    line_width: int,
    overwrite: bool,
    fs_lookup: Mapping[str, float],
) -> None:
    record_id = csv_path.stem
    image_path = image_dir / f"{record_id}.png"
    mask_path = mask_dir / f"{record_id}_mask.png"
    if not overwrite and image_path.exists() and mask_path.exists():
        return

    fs_override = fs_lookup.get(record_id)
    rng_seed = hash(record_id) & 0xFFFFFFFF
    image = render_ecg_page_from_csv(
        csv_path,
        grid_path,
        options=options,
        header_text="", # Disable default center header
        rng_seed=rng_seed,
        fs_override=fs_override,
    )
    
    # Add structured synthetic metadata
    image = add_synthetic_text(image, rng_seed)
    
    image_dir.mkdir(parents=True, exist_ok=True)
    image.save(image_path)

    mask_dir.mkdir(parents=True, exist_ok=True)
    render_wave_mask(
        csv_path,
        grid_path,
        mask_path,
        options=options,
        line_width=line_width,
        fs_override=fs_override,
        rng_seed=rng_seed,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-root", type=Path, default=GEN_SRC / "train", help="含有多份 ECG CSV 的目录（会递归查找）")
    ap.add_argument("--grid", type=Path, default=GEN_SRC / "export" / "clean_ecg_grid.png", help="网格 PNG")
    ap.add_argument("--output-dir", type=Path, default=Path("gen_dataset/wave"), help="兼容参数：若未指定 --image-dir/--mask-dir，则写入该目录")
    ap.add_argument("--image-dir", type=Path, default=None, help="整页 PNG 输出目录（默认与 --output-dir 相同）")
    ap.add_argument("--mask-dir", type=Path, default=None, help="波形掩模 PNG 输出目录（默认与整页目录相同）")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 份 CSV，用于测试")
    ap.add_argument("--overwrite", action="store_true", help="若已存在同名输出，是否覆盖")
    ap.add_argument("--cal-dx", type=float, default=1.0)
    ap.add_argument("--conn-w", type=int, default=6)
    ap.add_argument("--conn-h", type=int, default=55)
    ap.add_argument("--conn-dx", type=float, default=0.0)
    ap.add_argument("--gutter-boxes", type=float, default=None)
    ap.add_argument("--edge-margin-boxes", type=float, default=None)
    ap.add_argument("--first-lead-box", type=float, default=2.0)
    ap.add_argument("--row-top-boxes", type=str, default="18,25,32,39")
    ap.add_argument("--row-gap-boxes", type=float, default=0.0)
    ap.add_argument("--baseline-offset-boxes", type=float, default=0.0)
    ap.add_argument("--baseline-jitter-boxes", type=float, default=0.25)
    ap.add_argument("--start-mode", type=str, default="first_nonzero", choices=["start", "best", "sequential", "first_nonzero"])
    ap.add_argument("--start-sec", type=float, default=None)
    ap.add_argument("--nonzero-eps", type=float, default=1e-4)
    ap.add_argument("--label-font-size", type=int, default=36)
    ap.add_argument("--small-font-size", type=int, default=16)
    ap.add_argument("--line-width", type=int, default=1, help="掩模线条宽度（像素）")
    ap.add_argument("--metadata-csv", type=Path, default=GEN_SRC / "train.csv", help="记录采样率的元数据 CSV (id,fs,...)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    options = build_render_options(args)
    csv_files = list(iter_csv_files(args.csv_root))
    if args.limit is not None:
        csv_files = csv_files[: args.limit]
    fs_lookup = load_fs_lookup(args.metadata_csv)

    image_dir = args.image_dir or args.output_dir
    mask_dir = args.mask_dir or args.output_dir

    iterator = csv_files
    if tqdm is not None:
        iterator = tqdm(csv_files, desc="Rendering waveforms", unit="file")
    for idx, csv_path in enumerate(iterator, 1):
        render_single(
            csv_path,
            args.grid,
            image_dir,
            mask_dir,
            options,
            line_width=args.line_width,
            overwrite=args.overwrite,
            fs_lookup=fs_lookup,
        )


if __name__ == "__main__":
    main()
