import glob
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing

def check_file(path):
    try:
        with Image.open(path) as img:
            img.verify() # Fast check
        # Re-open to check if it can be loaded fully (verify doesn't decode)
        with Image.open(path) as img:
            img.load()
        return None
    except Exception as e:
        return f"{path}: {e}"

def main():
    root = Path("dataset_grid_wave_mix")
    images = sorted(list(root.glob("images/*.png")))
    masks = sorted(list(root.glob("masks/*.png")))
    
    all_files = images + masks
    print(f"Checking integrity of {len(all_files)} files...")
    
    corrupt_files = []
    
    with multiprocessing.Pool() as pool:
        for error in tqdm(pool.imap_unordered(check_file, all_files), total=len(all_files)):
            if error:
                corrupt_files.append(error)
                print(f"Corrupt: {error}")
                # Optional: Delete corrupt file so it can be re-generated
                # path = error.split(":")[0]
                # os.remove(path)

    if corrupt_files:
        print(f"\nFound {len(corrupt_files)} corrupt files.")
        print("You may want to delete them and re-run the build script.")
    else:
        print("\nAll files passed integrity check.")

if __name__ == "__main__":
    main()
