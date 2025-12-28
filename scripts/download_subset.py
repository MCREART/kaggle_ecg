import os
import random
import sys
from pathlib import Path
from collections import defaultdict

# Try importing kaggle
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Please install kaggle: pip install kaggle")
    sys.exit(1)

COMPETITION = "physionet-ecg-image-digitization"
FILE_LIST_PATH = "all_files.txt"
OUTPUT_ROOT = "downloaded_subset"

def main():
    if not os.path.exists(FILE_LIST_PATH):
        print(f"File list {FILE_LIST_PATH} not found. Please generate it first.")
        return

    print("Parsing file list...")
    id_to_files = defaultdict(list)
    
    with open(FILE_LIST_PATH, "r") as f:
        # Skip header if any (Kaggle CLI output often has headers)
        # The output format is usually: name, size, creationDate
        # We need the first column.
        lines = f.readlines()
        
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        filename = parts[0]
        
        # We look for train/<ID>/...
        if filename.startswith("train/") and "/" in filename:
            # Structure: train/ID/file
            seg = filename.split("/")
            if len(seg) >= 3:
                sample_id = seg[1]
                id_to_files[sample_id].append(filename)

    print(f"Found {len(id_to_files)} unique IDs.")
    
    if len(id_to_files) < 10:
        print("Not enough IDs found to sample 10. Downloading all.")
        selected_ids = list(id_to_files.keys())
    else:
        selected_ids = random.sample(list(id_to_files.keys()), 10)
        
    print(f"Selected 10 IDs: {selected_ids}")
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    # Download
    print("Starting download...")
    for sample_id in selected_ids:
        print(f"Processing {sample_id}...")
        files = id_to_files[sample_id]
        
        # Define local path
        local_dir = Path(OUTPUT_ROOT) / sample_id
        local_dir.mkdir(parents=True, exist_ok=True)
        
        for file_name in files:
            print(f"  Downloading {file_name} ...")
            # The file_name is like 'train/123/123.png'
            # We want to download it. Kaggle API 'competition_download_file' downloads the file.
            # It usually preserves the name or downloads to path.
            
            # Note: competition_download_file(competition, file_name, path=None, force=False, quiet=False)
            # If path is specified, it downloads there.
            try:
                # We want to put it in local_dir. 
                # But file_name has 'train/ID/' prefix. Kaggle might strip or keep it.
                # Usually it downloads the file to the path.
                api.competition_download_file(
                    COMPETITION,
                    file_name,
                    path=str(local_dir),
                    quiet=True
                )
                # Note: The API might download it as a zip if it's large, but individual files usually come as is.
                # However, sometimes it zips if it's a folder. Here we pass specific file.
                
                # Check if it was downloaded as a zip (Kaggle behavior quirks)
                # But usually single file download is fine.
            except Exception as e:
                print(f"  Error downloading {file_name}: {e}")

    print("Download complete.")
    print(f"Data saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
