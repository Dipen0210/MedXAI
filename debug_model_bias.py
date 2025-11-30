import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from src.inference import MedicalDeepfakeDetector
from src.scanner import FullScanDetector
import os

def get_center_patch(slice_path, scanner):
    if not os.path.exists(slice_path):
        print(f"File not found: {slice_path}")
        return None
        
    ds = pydicom.dcmread(slice_path)
    img_norm = scanner.normalize_slice(ds)
    
    h, w = img_norm.shape
    half = scanner.patch_size // 2
    cx, cy = w // 2, h // 2
    
    patch = img_norm[cy-half:cy+half, cx-half:cx+half]
    return patch

def check_patch(name, patch, detector):
    if patch is None: return
    
    from PIL import Image
    img_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    input_tensor = detector.transform(img_pil).unsqueeze(0).to(detector.device)
    
    detector.model.eval()
    with torch.no_grad():
        output = detector.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
    s0 = probs[0][0].item()
    s1 = probs[0][1].item()
    
    print(f"--- {name} ---")
    print(f"Class 0: {s0:.4f}")
    print(f"Class 1: {s1:.4f}")
    print(f"Predicted: Class {torch.argmax(probs).item()}")

def debug_bias():
    print("Debugging Model Bias...")
    model_path = "model.pth"
    scanner = FullScanDetector(model_path)
    detector = scanner.classifier
    
    # 1. Known Fake (Patient 1280, Slice 100)
    # We know the specific fake spot is at 323, 381
    fake_path = "Data/CT_Scans/EXP1_blind/1280/100.dcm"
    ds = pydicom.dcmread(fake_path)
    img_norm = scanner.normalize_slice(ds)
    target_x, target_y = 323, 381
    half = scanner.patch_size // 2
    fake_patch = img_norm[target_y-half:target_y+half, target_x-half:target_x+half]
    
    check_patch("KNOWN FAKE PATCH (1280)", fake_patch, detector)
    
    # 2. Known Real (Patient 1531 - TB, Slice 100)
    # Just take center patch
    real_path = "Data/CT_Scans/EXP1_blind/1531/100.dcm"
    real_patch = get_center_patch(real_path, scanner)
    check_patch("KNOWN REAL PATCH (1531)", real_patch, detector)

if __name__ == "__main__":
    debug_bias()
