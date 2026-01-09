#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def analyze_distribution(mask_dir: str, limit: int = None):
    mask_path = Path(mask_dir)
    files = list(mask_path.glob("*_mask.png"))
    
    if not files:
        print(f"No masks found in {mask_dir}")
        return

    if limit:
        files = files[:limit]
        
    print(f"Analyzing {len(files)} masks...")
    
    # Class mapping
    # 0: BG, 1: H-Line, 2: V-Line, 3: Intersection, 4: Wave
    class_names = {
        0: "Background",
        1: "H-Line",
        2: "V-Line",
        3: "Intersection",
        4: "Wave"
    }
    
    total_pixels = 0
    counts = defaultdict(int)
    
    for f in tqdm(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        unique, u_counts = np.unique(img, return_counts=True)
        total_pixels += img.size
        
        for val, count in zip(unique, u_counts):
            counts[val] += count
            
    print("\n=== Class Distribution ===")
    print(f"{'Class ID':<10} {'Name':<15} {'Pixels':<15} {'Ratio (%)'}")
    print("-" * 50)
    
    sorted_classes = sorted(list(counts.keys()))
    
    for cls in sorted_classes:
        count = counts[cls]
        ratio = (count / total_pixels) * 100
        name = class_names.get(cls, f"Unknown({cls})")
        print(f"{cls:<10} {name:<15} {count:<15} {ratio:.4f}%")
        
    print("-" * 50)
    
    # Suggest weights
    # A common heuristic is Inverted Frequency or something milder like 1/sqrt(freq)
    print("\n=== Suggested Simple Weights (Inverse Ratio) ===")
    base_ratio = counts.get(0, 1) / total_pixels # BG ratio
    for cls in sorted_classes:
        if cls == 0: continue
        if counts[cls] == 0:
            print(f"Class {cls}: N/A (0 pixels)")
            continue
        ratio = counts[cls] / total_pixels
        weight = base_ratio / ratio
        print(f"Class {cls} ({class_names.get(cls)}): {weight:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=str, default="dataset_grid_wave_mix/masks")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of images to analyze for speed")
    args = parser.parse_args()
    
    analyze_distribution(args.mask_dir, args.limit)
