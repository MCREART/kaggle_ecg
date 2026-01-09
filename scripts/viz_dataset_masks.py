
import os
import random
import numpy as np
import cv2
from pathlib import Path

def visualize_masks(mask_dir, output_dir, count=5):
    mask_dir = Path(mask_dir)
    # Assume image_dir is parallel to mask_dir
    image_dir = mask_dir.parent / "images"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    masks = list(mask_dir.glob("*_mask.png"))
    if not masks:
        print("No masks found!")
        return
        
    selected = random.sample(masks, min(len(masks), count))
    
    print(f"Visualizing {len(selected)} samples...")
    
    for p in selected:
        # Load Mask
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        
        # Determine Image Path: remove '_mask' from stem
        image_name = p.stem.replace("_mask", "") + ".png"
        image_path = image_dir / image_name
        
        # Load Image
        if image_path.exists():
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        else:
            print(f"[Warn] Image not found: {image_path}")
            # Create dummy black image if missing
            img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Create Colorized Mask
        h, w = mask.shape
        viz_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color mapping (BGR for OpenCV)
        # Color mapping (BGR for OpenCV)
        # 0: Background -> Black
        # 1: H-Line -> Red (0, 0, 255)
        # 2: V-Line -> Blue (255, 0, 0)
        # 3: Integration -> Yellow (0, 255, 255)
        # 4: Wave -> Green (0, 255, 0)
        
        viz_mask[mask == 1] = [0, 0, 255]      # Red
        viz_mask[mask == 2] = [255, 0, 0]      # Blue
        viz_mask[mask == 3] = [0, 255, 255]    # Yellow
        viz_mask[mask == 4] = [0, 255, 0]      # Green
        
        # Blend/Resize if needed (assuming sizes match)
        if img.shape[:2] != mask.shape:
             img = cv2.resize(img, (w, h))
             
        # Combine side-by-side
        combined = np.hstack([img, viz_mask])
        
        # Save
        out_path = output_dir / f"viz_{image_name}"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    visualize_masks("dataset_grid_wave_mix/masks", "viz_masks_output")
