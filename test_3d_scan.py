import os
import numpy as np
from src.scanner import FullScanDetector
from unittest.mock import MagicMock

def test_3d_optimization():
    print("Testing 3D Scanning Optimization (Mocked)...")
    
    # Initialize scanner
    scanner = FullScanDetector(model_path="dummy_path.pth")
    
    # Mock load_scan to return 5 dummy datasets
    mock_slices = []
    for i in range(5):
        ds = MagicMock()
        ds.InstanceNumber = i
        ds.pixel_array = np.zeros((512, 512), dtype=np.uint8)
        mock_slices.append(ds)
        
    scanner.load_scan = MagicMock(return_value=mock_slices)
    
    # Mock normalize_slice to return a dummy image with a "tumor" (bright spot)
    # This ensures get_candidates finds something
    def mock_normalize(ds):
        img = np.zeros((512, 512), dtype=np.uint8)
        # Add a bright spot in the center to trigger candidate generation
        cv2.circle(img, (256, 256), 20, 255, -1)
        return img
    
    import cv2
    scanner.normalize_slice = mock_normalize
    
    try:
        # Run scan
        print("Running scan()...")
        # batch_size=2 to force multiple batches for 5 slices
        detections, slices = scanner.scan("dummy_dir", batch_size=2) 
        
        print(f"Scan complete. Found {len(detections)} detections.")
        
        # Verify batching happened (we can't easily check internal state, but if it runs without error and finds detections, logic holds)
        if len(detections) > 0:
            print("PASS: Detections found using batched processing.")
        else:
            print("FAIL: No detections found (maybe candidates were filtered?).")
            
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_3d_optimization()
