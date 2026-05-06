import os
import pydicom
import numpy as np
import cv2
import torch
from src.inference import MedicalDeepfakeDetector

class FullScanDetector:
    def __init__(self, model_path, patch_size=64):
        self.classifier = MedicalDeepfakeDetector(model_path)
        self.patch_size = patch_size
        
    def load_scan(self, scan_dir):
        """Loads a DICOM series from a directory, sorted by instance number."""
        slices = []
        for s in os.listdir(scan_dir):
            if s.endswith(".dcm"):
                try:
                    ds = pydicom.dcmread(os.path.join(scan_dir, s))
                    slices.append(ds)
                except: pass
        slices.sort(key=lambda x: int(x.InstanceNumber))
        return slices

    def get_hu(self, ds):
        """Converts DICOM slice to raw Hounsfield Units."""
        img = ds.pixel_array.astype(np.float64)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        img = img * slope + intercept
        return img

    def normalize_slice(self, ds):
        """Converts DICOM slice to normalized 0-255 numpy array."""
        img = self.get_hu(ds)
        
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img[img > 1] = 1.
        img[img < 0] = 0.
        return (img * 255).astype(np.uint8)

    def get_candidates(self, img_norm, raw_img=None):
        """
        Fast candidate generation using OpenCV Contours.
        Returns list of (x, y) centers.
        """
        # Threshold to find bright spots (nodules/vessels)
        # Lung window is normalized, so bright spots are > 150-200 usually
        # For JPGs, we might need to be more lenient or ensure normalization is correct.
        # If input is already 0-255, we just use it.
        
        # HARD MASKING: Physically remove the table and edges
        # This is more robust than filtering centroids later
        # FIX: Work on a copy to avoid modifying the original image used for patch extraction
        img_masked = img_norm.copy()
        
        # OPTION 3: HU Filtering (Remove Bone)
        if raw_img is not None:
            # Bone is typically > 400-600 HU. 
            # We mask out anything > 600 to be safe (spine, ribs, pelvis).
            bone_mask = raw_img > 600
            img_masked[bone_mask] = 0
        
        h, w = img_masked.shape
        img_masked[int(h*0.80):, :] = 0  # Black out bottom 20% (Table) - Relaxed from 25%
        img_masked[:int(h*0.15), :] = 0  # Black out top 15% (Sternum/Anterior Wall)
        
        # Circular Mask to remove corner artifacts
        center = (int(w/2), int(h/2))
        radius = int(min(h, w) * 0.42) # TIGHTER MASK: Reduced from 0.48 to 0.42 to avoid edge artifacts
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
        
        # Morphological Erosion to clean up mask edges
        # Reduced to 3x3 kernel and 1 iteration to avoid deleting small nodules (like Patient 1901)
        mask_uint8 = (mask.astype(np.uint8) * 255)
        kernel = np.ones((3,3), np.uint8)
        mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        mask = mask_eroded > 0
        
        img_masked[~mask] = 0

        # Threshold to find bright spots (nodules/vessels)
        # Restoring to 130 (Balanced)
        _, thresh = cv2.threshold(img_masked, 130, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        # Helper to process valid contours
        def process_contour(cnt, offset=(0,0)):
            area = cv2.contourArea(cnt)
            if 50 < area < 15000: # Optmized to 50
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect_ratio = float(w_box)/h_box
                
                if 0.3 < aspect_ratio < 3.0:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"]) + offset[0]
                        cY = int(M["m01"] / M["m00"]) + offset[1]
                        
                        # FILTER: Ignore Table (Bottom 20%) and Sternum (Top 15%)
                        if cY > h * 0.80 or cY < h * 0.15: return
                        
                        candidates.append((cX, cY))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Standard check
            # OPTIMIZATION: Increased min area from 20 -> 50 to reduce noise candidates
            # The smallest fake nodule (Patient 1901) is ~289 pixels. 
            # 50 offers a safe margin while filtering out tiny specs.
            if 50 < area < 15000:
                process_contour(cnt)
                
            # Iterative Thresholding for large blobs (e.g. merged with lung)
            elif area >= 15000:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                roi = img_masked[y:y+h_box, x:x+w_box]
                
                # Re-threshold at higher value to separate nodules from lung/vessels
                _, sub_thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
                sub_contours, _ = cv2.findContours(sub_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                for sub_cnt in sub_contours:
                    process_contour(sub_cnt, offset=(x, y))
        
        return candidates

    def non_max_suppression(self, detections, iou_threshold=0.1):
        if not detections: return []
        
        # Convert to list of boxes [x1, y1, x2, y2, score, index]
        boxes = []
        half = self.patch_size // 2
        for i, d in enumerate(detections):
            boxes.append([d['x']-half, d['y']-half, d['x']+half, d['y']+half, d['score'], i])
            
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep_indices = []
        
        while boxes:
            current = boxes.pop(0)
            keep_indices.append(current[5])
            
            rest = []
            for box in boxes:
                # Calculate IoU
                xx1 = max(current[0], box[0])
                yy1 = max(current[1], box[1])
                xx2 = min(current[2], box[2])
                yy2 = min(current[3], box[3])
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                
                intersection = w * h
                area_current = (current[2] - current[0]) * (current[3] - current[1])
                area_box = (box[2] - box[0]) * (box[3] - box[1])
                union = area_current + area_box - intersection
                
                iou = intersection / union
                
                if iou < iou_threshold:
                    rest.append(box)
            boxes = rest
            
        return [detections[i] for i in keep_indices]



    def check_lung_context(self, img_norm, x, y, radius=20, threshold=100):
        """
        Checks if the candidate at (x,y) is likely inside the lung.
        Heuristic: Sample a ring/box around the candidate.
        If the surroundings are mostly dark (< threshold), it's in the lung/air.
        If surroundings are bright, it's in tissue/bone.
        """
        h, w = img_norm.shape
        
        # Define a ring/frame around the candidate
        # We look at a box from radius-5 to radius
        r_inner = 14 # Increased from 12 to avoid self-filtering large nodules
        r_outer = 26 # Increased slightly
        
        x1 = max(0, x - r_outer)
        x2 = min(w, x + r_outer)
        y1 = max(0, y - r_outer)
        y2 = min(h, y + r_outer)
        
        # Extract the patch
        patch = img_norm[y1:y2, x1:x2]
        if patch.size == 0: return False
        
        # Mask out the inner part (the nodule itself) to check ONLY context
        # Center of patch is roughly (x-x1, y-y1) -> (r_outer, r_outer)
        py, px = np.ogrid[:patch.shape[0], :patch.shape[1]]
        # Center relative to patch
        cy_p, cx_p = y - y1, x - x1
        
        dist_sq = (px - cx_p)**2 + (py - cy_p)**2
        mask = (dist_sq >= r_inner**2) # Only look outside r_inner
        
        surrounding_pixels = patch[mask]
        
        if surrounding_pixels.size == 0: return False
        
        mean_val = np.mean(surrounding_pixels)
        
        # If mean value is High (White), it's surrounded by tissue -> ROI is excluded
        # If mean value is Low (Black), it's surrounded by air -> Keep
        
        # Threshold: 
        # Air is ~0-30. Lung tissue is ~20-60. 
        # Solid Tissue is > 100.
        # Nodule is > 100.
        # Relaxed threshold to 130 to be very permissive for dense nodules
        
        return mean_val < 100


    def extract_cube_from_volume(self, volume_hu, center_idx, cx, cy, cube_size=32):
        """
        Extracts and normalizes a 3D cube to match training preprocessing.
        volume_hu: (D, H, W) in HU units
        Returns: (cube_size, cube_size, cube_size) normalized to [0,1]
        """
        max_d, max_h, max_w = volume_hu.shape
        half = cube_size // 2
        
        # Pad volume to avoid boundary issues
        pad = ((half, half), (half, half), (half, half))
        padded = np.pad(volume_hu, pad, mode='constant', constant_values=-1000)
        
        # Adjust coordinates for the padded volume
        z_center = center_idx + half
        x_center = cx + half  
        y_center = cy + half
        
        # Extract cube
        z1 = z_center - half
        z2 = z_center + half
        y1 = y_center - half
        y2 = y_center + half
        x1 = x_center - half
        x2 = x_center + half
        
        cube_hu = padded[z1:z2, y1:y2, x1:x2]
        
        # Normalize HU to [0, 1] (MATCH TRAINING)
        cube_norm = (cube_hu - (-1000)) / (400 - (-1000))
        cube_norm = np.clip(cube_norm, 0, 1)
        
        return cube_norm.astype(np.float32)

    def valiv_coords(self, x, y, h, w):
        return 0 <= x < w and 0 <= y < h

    def scan(self, scan_dir, sensitivity=0.6, batch_size=32, slice_step=2, progress_callback=None):
        """Scans the full 3D volume using optimized Batch Processing."""
        slices = self.load_scan(scan_dir)
        if not slices: return [], []

        all_detections = []

        print(f"Scanning {len(slices)} slices (step={slice_step})...")

        # Pre-load the entire volume to RAM
        print("Pre-loading 3D Volume (HU units)...")
        volume_hu = []
        for ds in slices:
            volume_hu.append(self.get_hu(ds))

        volume_hu = np.stack(volume_hu)  # (D, H, W) in HU units
        print(f"Volume Shape: {volume_hu.shape}")

        # Buffers for batching
        cube_buffer = []
        meta_buffer = []

        # Temporary storage for detections before NMS
        raw_detections_by_slice = {i: [] for i in range(len(slices))}

        active_indices = list(range(0, len(slices), slice_step))
        total = len(active_indices)

        for step, i in enumerate(active_indices):
            ds = slices[i]
            if progress_callback:
                progress_callback(step / total, f"Scanning slice {i+1}/{len(slices)}...")
            if i % 10 == 0:
                print(f"Processing slice {i}/{len(slices)}...")
            
            # Use pre-loaded HU volume for both candidate detection and cube extraction
            raw_img = volume_hu[i]
            img_norm = self.normalize_slice(ds)  # For candidate detection only 
            

            # Get candidates on this slice
            candidates = self.get_candidates(img_norm, raw_img=raw_img)
            
            # --- NEW: Context Filter ---
            # User requirement: Nodules must be INSIDE the lung (surrounded by black air).
            # Filter out candidates that are surrounded by tissue (mediastinum, chest wall).
            valid_candidates = []
            for (x, y) in candidates:
                if self.check_lung_context(img_norm, x, y):
                    valid_candidates.append((x, y))
            candidates = valid_candidates
            
            print(f"Slice {i}: Found {len(candidates)} candidates")

            
            # Filter candidates: Ensure they are "inside" boundaries (handled by Mask in get_candidates)
            # User note: "nodules will be found only inside the boudaries... inside two left and right lungs in white marks"
            # Our existing get_candidates uses threshold > 130 (white marks) and mask (boundaries).
            
            for (x, y) in candidates:
                # Extract 32x32x32 cube with HU normalization (MATCH TRAINING!)
                cube = self.extract_cube_from_volume(volume_hu, i, x, y, cube_size=32)
                
                cube_buffer.append(cube)
                meta_buffer.append((i, x, y))
                
            # Process buffer if full
            while len(cube_buffer) >= batch_size:
                print(f"Batch inference on {len(cube_buffer[:batch_size])} items...")
                batch = cube_buffer[:batch_size]
                meta = meta_buffer[:batch_size]
                cube_buffer = cube_buffer[batch_size:]
                meta_buffer = meta_buffer[batch_size:]
                
                results = self.classifier.predict_batch(batch)
                
                for j, (label, score, _) in enumerate(results):
                    if score > sensitivity:
                        s_idx, px, py = meta[j]
                        # Skip heatmap generation here (causes OpenCV errors for 3D)
                        # App will regenerate heatmaps when needed
                        
                        raw_detections_by_slice[s_idx].append({
                            'slice': s_idx,
                            'x': px,
                            'y': py,
                            'label': label,
                            'score': score,
                            'heatmap': None  # Will be generated by app.py on demand
                        })
                        
        # Process remaining items in buffer
        if cube_buffer:
            results = self.classifier.predict_batch(cube_buffer)
            for j, (label, score, _) in enumerate(results):
                if score > sensitivity:
                    s_idx, px, py = meta_buffer[j]
                    
                    raw_detections_by_slice[s_idx].append({
                        'slice': s_idx,
                        'x': px,
                        'y': py,
                        'label': label,
                        'score': score,
                        'heatmap': None  # Will be generated by app.py on demand
                    })
                    
        # Apply NMS per slice and aggregate
        # NOTE: For 3D verification, we might want 3D NMS, but per-slice is a good start
        # to identify which slice has the evidence.
        for i in range(len(slices)):
            dets = raw_detections_by_slice[i]
            if dets:
                dets = self.non_max_suppression(dets)
                for d in dets:
                    d['instance_number'] = slices[i].InstanceNumber
                    all_detections.append(d)
        
        # --- NEW: 3D Consistency Check ---
        all_detections = self.apply_3d_consistency(all_detections, dist_thresh=50, min_slices=2)

        if progress_callback:
            progress_callback(1.0, "Scan complete!")

        # Store volume_hu on self so app.py can generate heatmaps on demand
        self._last_volume_hu = volume_hu

        return all_detections, slices

    def generate_heatmaps_for_slice(self, detections_on_slice):
        """Generate Grad-CAM heatmaps only for detections on a specific slice (called on demand)."""
        volume_hu = getattr(self, '_last_volume_hu', None)
        if volume_hu is None:
            return
        for d in detections_on_slice:
            if d['heatmap'] is None:
                cube = self.extract_cube_from_volume(volume_hu, d['slice'], d['x'], d['y'], cube_size=32)
                _, _, heatmap = self.classifier.predict(cube)
                d['heatmap'] = heatmap

    def _group_by_3d_consistency(self, detections, dist_thresh, min_slices):
        """Groups detections spatially across slices. Returns only groups with >= min_slices."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x['slice'])
        n = len(detections)
        parent = list(range(n))

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(n):
            for j in range(i + 1, n):
                slice_diff = abs(detections[i]['slice'] - detections[j]['slice'])
                if slice_diff > 3:
                    break
                if slice_diff > 0:
                    dist = np.sqrt((detections[i]['x'] - detections[j]['x'])**2 +
                                   (detections[i]['y'] - detections[j]['y'])**2)
                    if dist < dist_thresh:
                        union(i, j)

        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(detections[i])

        verified = []
        for group in groups.values():
            if len(set(d['slice'] for d in group)) >= min_slices:
                verified.extend(group)
        return verified

    def apply_3d_consistency(self, detections, dist_thresh=50, min_slices=2):
        """
        Both FM and TM require 3D consistency (appearing in >= min_slices) to be kept.
        Single-slice detections are discarded as noise — real tampered nodules and
        real cancer nodules both span multiple consecutive slices.
        """
        if not detections:
            return []

        fake_detections = [d for d in detections if d['label'] == 'Fake']
        real_detections = [d for d in detections if d['label'] == 'Real']

        verified_fake = self._group_by_3d_consistency(fake_detections, dist_thresh, min_slices)
        for d in verified_fake:
            d['label'] = 'FM'

        verified_real = self._group_by_3d_consistency(real_detections, dist_thresh, min_slices)
        for d in verified_real:
            d['label'] = 'TM'
            d['score'] = 0.99

        return verified_fake + verified_real


