import cv2
import numpy as np
import glob
import os

def create_overlay():
    image_files = sorted(glob.glob("final_dataset_test/images/*.png"))
    mask_files = sorted(glob.glob("final_dataset_test/masks/*.png"))
    os.makedirs("visualizations", exist_ok=True)

    print(f"Found {len(image_files)} sample pairs.")

    for img_path, mask_path in zip(image_files, mask_files):
        # Double check pairing
        if os.path.basename(img_path).replace(".png", "") != os.path.basename(mask_path).replace("_mask.png", ""):
            print(f"Mismatch: {img_path} vs {mask_path}")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create localized colored overlays
        # Grid (1) -> Blue
        # Wave (2) -> Red
        
        overlay = img.copy()
        
        # Grid: Blue
        overlay[mask == 1] = [255, 0, 0] # BGR
        
        # Wave: Red
        overlay[mask == 2] = [0, 0, 255] # BGR

        # Blend
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        basename = os.path.basename(img_path)
        out_path = f"visualizations/vis_{basename}"
        cv2.imwrite(out_path, img)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    create_overlay()
