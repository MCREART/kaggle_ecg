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
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    print("Fetching file list via API (handling pagination)...")
    id_to_files = defaultdict(list)
    unique_ids = set()
    needed_count = 50  # Target number of samples
    
    # Iterate pages usually implicitly or manually. 
    # For competition_list_files, valid return is list of File objects.
    # We loop until we have enough IDs.
    
    MAX_PAGES = 50
    page_count = 0
    token = None
    
    while len(unique_ids) < needed_count and page_count < MAX_PAGES:
        print(f"Fetching page {page_count+1} (token={token if token else 'None'})...")
        try:
            kwargs = {}
            if token:
                kwargs['page_token'] = token
            files_resp = api.competition_list_files(COMPETITION, **kwargs)
        except Exception as e:
            print(f"Error fetching page: {e}")
            break
            
        # Handle ApiListDataFilesResponse or list
        if hasattr(files_resp, 'files'):
            files_list = files_resp.files
        else:
            files_list = files_resp
            
        if not files_list:
            print("No more files returned.")
            break
            
        print(f"  Page returned {len(files_list)} items.")
        
        for f in files_list:
            filename = str(f)
            if hasattr(f, 'name'):
                filename = f.name
                
            if filename.startswith("train/") and "/" in filename:
                seg = filename.split("/")
                if len(seg) >= 3:
                    sample_id = seg[1]
                    id_to_files[sample_id].append(filename)
                    unique_ids.add(sample_id)
        
        print(f"  Currently have {len(unique_ids)} unique IDs.")
        
        # Update token for next page
        # Try common attribute names for next page token
        token = None
        if hasattr(files_resp, 'nextPageToken') and files_resp.nextPageToken:
             token = files_resp.nextPageToken
        elif hasattr(files_resp, 'next_page_token') and files_resp.next_page_token:
             token = files_resp.next_page_token
             
        if not token:
             print("No next page token found. End of list.")
             break
             
        page_count += 1
    
    if len(unique_ids) < needed_count:
        print(f"Not enough IDs found to sample {needed_count}. Downloading all {len(unique_ids)}.")
        selected_ids = list(unique_ids)
    else:
        selected_ids = random.sample(list(unique_ids), needed_count)
        
    print(f"Selected {len(selected_ids)} IDs: {selected_ids}")
    
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
