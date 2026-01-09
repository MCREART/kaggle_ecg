#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Import shared detection logic
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.crop_generator.grid_utils import detect_grid

def main():
    parser = argparse.ArgumentParser(description="Pre-compute semantic grid masks (H/V/Inter)")
    parser.add_argument("--grid-image", type=Path, required=True, help="Original red grid image (for detection)")
    parser.add_argument("--mask-image", type=Path, default=None, help="Binary grid mask (optional, for verification)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save separated masks")
    args = parser.parse_args()

    # 1. Detect Grid on the colored image (most robust)
    print(f"Detecting grid from: {args.grid_image}")
    try:
        grid_spec = detect_grid(args.grid_image)
    except Exception as e:
        print(f"Error: Detection failed: {e}")
        return

    # 2. Create Masks
    w, h = grid_spec.image.size
    mask_h = np.zeros((h, w), dtype=np.uint8)
    mask_v = np.zeros((h, w), dtype=np.uint8)
    
    line_width = 3  # Reverted to 3px for robustness (thinned later)
    
    print(f"Generating components: {len(grid_spec.y_lines)} H-lines, {len(grid_spec.x_lines)} V-lines")

    # Draw Horizontal
    for y_coord in grid_spec.y_lines:
        y1 = max(0, int(y_coord - line_width // 2))
        y2 = min(h, int(y_coord + line_width // 2) + 1)
        mask_h[y1:y2, :] = 255

    # Draw Vertical
    for x_coord in grid_spec.x_lines:
        x1 = max(0, int(x_coord - line_width // 2))
        x2 = min(w, int(x_coord + line_width // 2) + 1)
        mask_v[:, x1:x2] = 255

    # Intersections
    mask_i = cv2.bitwise_and(mask_h, mask_v)

    # 3. Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    h_path = args.output_dir / "grid_h_mask.png"
    v_path = args.output_dir / "grid_v_mask.png"
    i_path = args.output_dir / "grid_i_mask.png"
    
    cv2.imwrite(str(h_path), mask_h)
    cv2.imwrite(str(v_path), mask_v)
    cv2.imwrite(str(i_path), mask_i)
    
    print(f"Saved masks to {args.output_dir}:")
    print(f"  - {h_path.name}")
    print(f"  - {v_path.name}")
    print(f"  - {i_path.name}")

if __name__ == "__main__":
    main()
