import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from src.inference import MedicalDeepfakeDetector
from src.scanner import FullScanDetector

def debug_fake_slice():
    print("Debugging Fake Slice 1280 - Slice 100")
    
    # Path to known fake slice
    slice_path = "Data/CT_Scans/EXP1_blind/1280/100.dcm"
    model_path = "model.pth"
    
    # Known fake coordinates from labels_exp1.csv
    # FB,1280,100,323,381
    target_x = 323
    target_y = 381
    
    # Initialize
    scanner = FullScanDetector(model_path)
    detector = scanner.classifier
    
    # Load and Normalize
    ds = pydicom.dcmread(slice_path)
    img_norm = scanner.normalize_slice(ds)
    
    print(f"Image Shape: {img_norm.shape}, Max: {img_norm.max()}, Min: {img_norm.min()}")
    
    # Extract Patch
    half = scanner.patch_size // 2
    x1 = target_x - half
    x2 = target_x + half
    y1 = target_y - half
    y2 = target_y + half
    
    patch = img_norm[y1:y2, x1:x2]
    print(f"Patch Shape: {patch.shape}")
    print(f"Patch Max Value: {patch.max()}")
    print(f"Patch Mean Value: {patch.mean()}")
    
    if patch.shape != (64, 64):
        print("Error: Patch size mismatch")
        return

    # Run Inference
    # We need to manually preprocess to check raw scores
    from PIL import Image
    img_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    input_tensor = detector.transform(img_pil).unsqueeze(0).to(detector.device)
    
    detector.model.eval()
    with torch.no_grad():
        output = detector.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
    score_0 = probs[0][0].item()
    score_1 = probs[0][1].item()
    
    print(f"\nRaw Model Output:")
    print(f"Class 0 (Fake?): {score_0:.4f}")
    print(f"Class 1 (Real?): {score_1:.4f}")
    
    if score_0 > score_1:
        print("Model predicts Class 0")
    else:
        print("Model predicts Class 1")
        
    # Check what 'predict' returns
    label, score, _ = detector.predict(patch)
    print(f"\nDetector.predict() returns: {label} with score {score:.4f}")
    
    # Check if candidate generation even finds it
    print("\n--- Debugging Candidate Generation ---")
    
    # Replicate get_candidates logic with Threshold Sweep
    img_masked = img_norm.copy()
    h, w = img_masked.shape
    img_masked[int(h*0.80):, :] = 0  # Black out bottom 20%
    
    center = (int(w/2), int(h/2))
    radius = int(min(h, w) * 0.48)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    img_masked[~mask] = 0
    
    # Test Adaptive Thresholding
    print("\n--- Testing Adaptive Thresholding ---")
    img_masked = img_norm.copy()
    h, w = img_masked.shape
    img_masked[int(h*0.80):, :] = 0
    
    center = (int(w/2), int(h/2))
    radius = int(min(h, w) * 0.48)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    img_masked[~mask] = 0
    
    # Test Iterative Thresholding
    print("\n--- Testing Iterative Thresholding ---")
    img_masked = img_norm.copy()
    h, w = img_masked.shape
    img_masked[int(h*0.80):, :] = 0
    
    center = (int(w/2), int(h/2))
    radius = int(min(h, w) * 0.48)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    img_masked[~mask] = 0
    
    _, thresh = cv2.threshold(img_masked, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 15000:
            final_candidates.append(cnt)
        elif area >= 15000:
            print(f"  -> Found large contour (Area: {area}). Re-thresholding ROI...")
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            roi = img_masked[y:y+h_box, x:x+w_box]
            
            # Re-threshold at higher value (e.g. 180)
            _, sub_thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
            sub_contours, _ = cv2.findContours(sub_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            print(f"     -> Found {len(sub_contours)} sub-contours.")
            for sub_cnt in sub_contours:
                # Adjust coordinates back to global
                sub_cnt += np.array([x, y])
                sub_area = cv2.contourArea(sub_cnt)
                if 20 < sub_area < 15000:
                    final_candidates.append(sub_cnt)
                    
    print(f"Total Final Candidates: {len(final_candidates)}")
    
    found_iterative = False
    if final_candidates:
        dists = []
        for cnt in final_candidates:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                cX, cY = x + w_box//2, y + h_box//2
            
            dist = np.sqrt((cX - target_x)**2 + (cY - target_y)**2)
            area = cv2.contourArea(cnt)
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect_ratio = float(w_box)/h_box
            dists.append((dist, area, aspect_ratio, cX, cY))
        
        dists.sort(key=lambda x: x[0])
        closest = dists[0]
        print(f"  -> Closest Contour: Dist={closest[0]:.2f}, Area={closest[1]}, AR={closest[2]:.2f}, Center=({closest[3]}, {closest[4]})")
        
        if closest[0] < 20:
            print("     -> MATCH FOUND!")
            if 20 < closest[1] < 15000 and 0.3 < closest[2] < 3.0 and closest[4] <= h * 0.80:
                print("     -> PASSED ALL FILTERS!")
                found_iterative = True
            else:
                if not (20 < closest[1] < 15000): print(f"     -> FAILED Area ({closest[1]})")
                if not (0.3 < closest[2] < 3.0): print(f"     -> FAILED AR ({closest[2]:.2f})")
                if closest[4] > h * 0.80: print(f"     -> FAILED Table ({closest[4]})")
    else:
        print("  -> No contours found.")
        
    if found_iterative:
        print("*** SUCCESS with Iterative Thresholding ***")

if __name__ == "__main__":
    debug_fake_slice()
