import pydicom
import numpy as np
from src.scanner import FullScanDetector

def verify_fix():
    print("Verifying Fix for Slice 1280 - Slice 100")
    
    slice_path = "Data/CT_Scans/EXP1_blind/1280/100.dcm"
    model_path = "model.pth"
    target_x = 323
    target_y = 381
    
    scanner = FullScanDetector(model_path)
    
    ds = pydicom.dcmread(slice_path)
    raw_img = scanner.get_hu(ds)
    img_norm = scanner.normalize_slice(ds)
    
    # Test 2D Mode (simulate missing HU)
    print("Testing in 2D Mode (raw_img=None)...")
    candidates = scanner.get_candidates(img_norm, raw_img=None)
    print(f"Found {len(candidates)} candidates.")
    
    found = False
    half = scanner.patch_size // 2
    
    for cx, cy in candidates:
        dist = np.sqrt((cx - target_x)**2 + (cy - target_y)**2)
        
        # Extract patch to check intensity
        if cy-half >= 0 and cy+half <= img_norm.shape[0] and cx-half >= 0 and cx+half <= img_norm.shape[1]:
            patch = img_norm[cy-half:cy+half, cx-half:cx+half]
            mean_val = patch.mean()
            max_val = patch.max()
        else:
            mean_val = 0
            max_val = 0
            
        print(f"Candidate: ({cx}, {cy}), Dist: {dist:.2f}, Mean: {mean_val:.1f}, Max: {max_val}")
        
        if dist < 30: # Relaxed distance check
            print("  -> MATCH FOUND! (Target Nodule)")
            found = True
            
    if found:
        print("\nSUCCESS: Target detected!")
    else:
        print("\nFAILURE: Target still missed.")

if __name__ == "__main__":
    verify_fix()
