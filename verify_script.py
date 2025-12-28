import numpy as np
import cv2
import os
import shutil
import subprocess
import glob
from PIL import Image

def setup_data():
    if os.path.exists("_test_verif"):
        shutil.rmtree("_test_verif")
    os.makedirs("_test_verif/wave", exist_ok=True)
    os.makedirs("_test_verif/ori_data", exist_ok=True)
    os.makedirs("_test_verif/output", exist_ok=True)

    # 1. Grid Mask (Static) - 1000x1000
    grid_mask = np.zeros((1000, 1000), dtype=np.uint8)
    for i in range(0, 1000, 100):
        cv2.line(grid_mask, (i, 0), (i, 1000), 255, 5)
        cv2.line(grid_mask, (0, i), (1000, i), 255, 5)
    cv2.imwrite("_test_verif/mask.png", grid_mask)

    # 2. Wave Image + Mask
    wave_img = np.full((1000, 1000, 3), 240, dtype=np.uint8)
    cv2.imwrite("_test_verif/wave/sample01.png", wave_img)

    # Wave Mask: Draw a thick diagonal line
    wave_mask = np.zeros((1000, 1000), dtype=np.uint8)
    cv2.line(wave_mask, (100, 100), (900, 900), 255, 30) # Thick enough to ensure coverage overlap
    cv2.imwrite("_test_verif/wave/sample01_mask.png", wave_mask)
    
    # 3. Dummy 'complete_mask.png' for default args if needed (though we pass --mask)
    # The script uses it from input arg.

def run_build():
    cmd = [
        "python", "scripts/build_dataset.py",
        "--input-root", "_test_verif/ori_data", # Empty but required
        "--mask", "_test_verif/mask.png",
        "--wave-root", "_test_verif/wave",       
        "--output-root", "_test_verif/output",
        "--samples-per-image", "1",
        "--negative-per-image", "0",
        "--limit", "1",
        "--wave-min-coverage", "0.0", # Force accept small coverage
        "--crop-out", "256",
        "--overwrite"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def verify_output():
    mask_files = glob.glob("_test_verif/output/masks/*.png")
    print("Generated masks:", mask_files)
    if not mask_files:
        print("FAILED: No masks generated")
        return

    for f in mask_files:
        mask = np.array(Image.open(f))
        uniques = np.unique(mask)
        print(f"File {f} Unique values: {uniques}")
        if 2 in uniques:
            print("SUCCESS: Found Waveform class (2)")
        else:
            print("WARNING: Waveform class (2) missing in this crop (might be random crop miss, but with large line it should differ)")
        
        if 1 in uniques:
             print("SUCCESS: Found Grid class (1)")
        
        if 2 in uniques and 1 in uniques:
            print("VERIFICATION PASSED: Both classes present.")
            return

    if len(mask_files) > 0:
        print("Completed check of all files.")

if __name__ == "__main__":
    setup_data()
    try:
        run_build()
        verify_output()
    except Exception as e:
        print("Error:", e)
