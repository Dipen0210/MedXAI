import os
import pandas as pd
import pydicom
import glob

DATA_DIR = "/Users/dipen/MVproject/Data"
CSV_PATH = os.path.join(DATA_DIR, "labels_exp1.csv")
CT_SCANS_DIR = os.path.join(DATA_DIR, "CT_Scans", "EXP1_blind")

def verify_data():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    
    # Pick a random row
    row = df.iloc[0]
    uuid = str(row['uuid'])
    slice_idx = int(row['slice'])
    
    scan_dir = os.path.join(CT_SCANS_DIR, uuid)
    if not os.path.exists(scan_dir):
        print(f"Error: Scan directory not found for UUID {uuid}")
        return

    print(f"Checking UUID: {uuid}, Slice Index from CSV: {slice_idx}")
    
    # List files
    dicom_files = glob.glob(os.path.join(scan_dir, "*.dcm"))
    dicom_files.sort() # Ensure sorted order if filenames are numbered
    
    print(f"Found {len(dicom_files)} DICOM files in {scan_dir}")
    
    # Check if a file with the slice index exists directly
    direct_file = os.path.join(scan_dir, f"{slice_idx}.dcm")
    if os.path.exists(direct_file):
        print(f"SUCCESS: Found file matching slice index: {slice_idx}.dcm")
    else:
        print(f"WARNING: No file named {slice_idx}.dcm. Checking file content for InstanceNumber...")
        
        # Check InstanceNumber in a few files to see the pattern
        for fpath in dicom_files[:5]:
            ds = pydicom.dcmread(fpath)
            print(f"File: {os.path.basename(fpath)}, InstanceNumber: {ds.InstanceNumber}")

if __name__ == "__main__":
    verify_data()
