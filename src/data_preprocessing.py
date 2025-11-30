import os
import pandas as pd
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

# Configuration
DATA_DIR = "/Users/dipen/MVproject/Data"
OUTPUT_DIR = "/Users/dipen/MVproject/processed_data"
CSV_PATH = os.path.join(DATA_DIR, "labels_exp1.csv")
CT_SCANS_DIR = os.path.join(DATA_DIR, "CT_Scans", "EXP1_blind")
PATCH_SIZE = 64

def normalize_hu(image):
    # Lung windowing
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return (image * 255).astype(np.uint8)

def get_slice_file(scan_dir, slice_idx):
    # Try direct match first
    p = os.path.join(scan_dir, f"{slice_idx}.dcm")
    if os.path.exists(p):
        return p
    
    # Fallback: search by InstanceNumber (slower but safer)
    # This is a simplified fallback; for speed, we might want to map all files first
    # But for now let's assume the filename matches the slice index usually
    return None

def extract_patches():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Create output directories
    classes = ['Real', 'Fake']
    for c in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)
    
    print(f"Processing {len(df)} labels...")
    
    success_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        uuid = str(row['uuid'])
        slice_idx = int(row['slice'])
        x = int(row['x'])
        y = int(row['y'])
        label_type = row['type'] # TB, TM, FB, FM
        
        # Map label types to Real/Fake
        # TM (True Malicious) -> Real Cancer
        # FM (False Malicious) -> Fake Cancer (Injected)
        # FB (False Benign) -> Fake Healthy (Removed Cancer) - This is tricky. 
        #   If we want to detect "Tampering", FB is also "Fake".
        #   If we want to detect "Cancer", FB is "Healthy".
        #   The user asked: "is it fake or real". Usually implies "Is this image tampered?"
        #   Let's define:
        #   Real = TM (Real Cancer)
        #   Fake = FM (Injected Cancer)
        #   We will ignore TB (Healthy) and FB (Removed) for now to keep it simple: "Real Cancer vs Fake Cancer"
        #   Or should we include everything?
        #   Let's stick to the most obvious: Real Cancer (TM) vs Fake Cancer (FM).
        
        if label_type == 'TM':
            category = 'Real'
        elif label_type == 'FM':
            category = 'Fake'
        else:
            continue # Skip TB and FB for this specific binary classifier
            
        scan_dir = os.path.join(CT_SCANS_DIR, uuid)
        dcm_path = get_slice_file(scan_dir, slice_idx)
        
        if not dcm_path or not os.path.exists(dcm_path):
            # Try finding by listing (handling potential leading zeros or different naming)
            # This is a simple heuristic
            continue

        try:
            ds = pydicom.dcmread(dcm_path)
            img = ds.pixel_array
            
            # Rescale Intercept/Slope if present
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
            img = img * slope + intercept
            
            img_norm = normalize_hu(img)
            
            # Extract patch
            half_size = PATCH_SIZE // 2
            
            # Check bounds
            if y - half_size < 0 or y + half_size > img_norm.shape[0] or \
               x - half_size < 0 or x + half_size > img_norm.shape[1]:
                continue
                
            patch = img_norm[y-half_size:y+half_size, x-half_size:x+half_size]
            
            # Save
            out_name = f"{uuid}_{slice_idx}_{x}_{y}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, category, out_name), patch)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {uuid} slice {slice_idx}: {e}")
            continue

    print(f"Successfully extracted {success_count} patches.")

if __name__ == "__main__":
    extract_patches()
