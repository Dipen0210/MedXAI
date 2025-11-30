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
        
        # Circular Mask to remove corner artifacts
        center = (int(w/2), int(h/2))
        radius = int(min(h, w) * 0.48) # Keep central 96%
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= radius
        img_masked[~mask] = 0

        # Threshold to find bright spots (nodules/vessels)
        # Restoring to 130 (Balanced)
        _, thresh = cv2.threshold(img_masked, 130, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        # Helper to process valid contours
        def process_contour(cnt, offset=(0,0)):
            area = cv2.contourArea(cnt)
            if 20 < area < 15000:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect_ratio = float(w_box)/h_box
                
                if 0.3 < aspect_ratio < 3.0:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"]) + offset[0]
                        cY = int(M["m01"] / M["m00"]) + offset[1]
                        
                        # FILTER: Ignore Table (Bottom 20%)
                        if cY > h * 0.80: return
                        
                        # FILTER: Intensity (For 2D mode where HU is missing)
                        # Remove very bright objects (Bone/Contrast)
                        if raw_img is None:
                            # Create a mask for this contour
                            mask = np.zeros(img_masked.shape, np.uint8)
                            # We need to shift contour by offset if it's a sub-contour
                            cnt_shifted = cnt + np.array(offset)
                            cv2.drawContours(mask, [cnt_shifted], -1, 255, -1)
                            mean_val = cv2.mean(img_masked, mask=mask)[0]
                            
                            # print(f"DEBUG: Contour Mean: {mean_val:.2f} at ({cX}, {cY})")
                            
                            # Bone is usually > 220-250 in normalized image
                            # Nodule contour is around 180-190
                            if mean_val > 200: 
                                # print(f"DEBUG: Rejected Bone/Contrast (Mean {mean_val:.2f})")
                                return
                        
                        candidates.append((cX, cY))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Standard check
            if 20 < area < 15000:
                process_contour(cnt)
                
            # Iterative Thresholding for large blobs (e.g. merged with lung)
            elif area >= 15000:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                roi = img_masked[y:y+h_box, x:x+w_box]
                
                # Re-threshold at higher value to separate nodules from lung/vessels
                _, sub_thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
                sub_contours, _ = cv2.findContours(sub_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for sub_cnt in sub_contours:
                    process_contour(sub_cnt, offset=(x, y))
        
        return candidates
        # print(f"DEBUG: Kept {len(candidates)} candidates after filtering.")
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

    def scan_slice(self, img_norm, slice_idx=0, sensitivity=0.7, raw_img=None):
        """Scans a single 2D image."""
        candidates = self.get_candidates(img_norm, raw_img=raw_img)
        if not candidates: return []
        
        patches = []
        valid_candidates = []
        
        half = self.patch_size // 2
        
        for (x, y) in candidates:
            if y-half < 0 or y+half > img_norm.shape[0] or x-half < 0 or x+half > img_norm.shape[1]:
                continue
            patch = img_norm[y-half:y+half, x-half:x+half]
            patches.append(patch)
            valid_candidates.append((x, y))
            
        if not patches: return []
        
        # Batch Prediction
        results = self.classifier.predict_batch(patches)
        
        detections = []
        for i, (label, score, _) in enumerate(results):
            if score > sensitivity:
                x, y = valid_candidates[i]
                
                # Re-run single inference to get Heatmap (Quality)
                # Optimization: Pass numpy array directly
                # Optimization 2: Only generate heatmap if we are going to keep it (already checked score > sensitivity)
                # Note: predict() returns heatmap, but predict_batch() does not.
                # We need to call predict() to get the heatmap.
                
                _, _, heatmap = self.classifier.predict(patches[i])
                
                detections.append({
                    'slice': slice_idx,
                    'x': x,
                    'y': y,
                    'label': label,
                    'score': score,
                    'heatmap': heatmap
                })
        
        # Apply Non-Maximum Suppression
        detections = self.non_max_suppression(detections)
        
        return detections

    def scan(self, scan_dir, sensitivity=0.75, batch_size=64):
        """Scans the full 3D volume using optimized Batch Processing."""
        slices = self.load_scan(scan_dir)
        all_detections = []
        
        print(f"Scanning {len(slices)} slices...")
        
        # Buffers for batching
        patch_buffer = []
        meta_buffer = [] # (slice_idx, x, y)
        
        # Temporary storage for detections before NMS
        raw_detections_by_slice = {i: [] for i in range(len(slices))}
        
        # Pre-load HU grids for 3D consistency (Optimization: Load on demand or cache?)
        # For now, we load row by row.
        
        for i, ds in enumerate(slices):
            if i % 10 == 0:
                print(f"Processing slice {i}/{len(slices)}...")
                
            raw_img = self.get_hu(ds)
            img_norm = self.normalize_slice(ds)
            candidates = self.get_candidates(img_norm, raw_img=raw_img)
            
            # OPTION 2: 3D Consistency Check
            # Check if this candidate exists in adjacent slices
            valid_candidates_3d = []
            for (x, y) in candidates:
                # Check neighbors (i-1, i+1)
                # We check if the HU value at (x,y) in neighbors is > -600 (Tissue/Bone/Fluid)
                # If it's Air (-1000), it's likely noise/artifact.
                
                consistent = False
                neighbors = []
                if i > 0: neighbors.append(i-1)
                if i < len(slices) - 1: neighbors.append(i+1)
                
                if not neighbors: # Single slice case
                    consistent = True
                else:
                    for n_idx in neighbors:
                        # We need to load the neighbor slice
                        # NOTE: This might be slow if we re-read DICOM. 
                        # But 'slices' list contains pydicom objects which lazy load pixel_array.
                        # To be efficient, we should probably cache, but let's try direct access first.
                        n_ds = slices[n_idx]
                        # Quick HU check without full normalization
                        slope = getattr(n_ds, 'RescaleSlope', 1)
                        intercept = getattr(n_ds, 'RescaleIntercept', 0)
                        val = n_ds.pixel_array[y, x] * slope + intercept
                        
                        if val > -600: # Threshold for "Something is there"
                            consistent = True
                            break
                
                if consistent:
                    valid_candidates_3d.append((x, y))
            
            candidates = valid_candidates_3d
            
            half = self.patch_size // 2
            for (x, y) in candidates:
                if y-half < 0 or y+half > img_norm.shape[0] or x-half < 0 or x+half > img_norm.shape[1]:
                    continue
                patch = img_norm[y-half:y+half, x-half:x+half]
                patch_buffer.append(patch)
                meta_buffer.append((i, x, y))
                
            # Process buffer if full
            while len(patch_buffer) >= batch_size:
                batch = patch_buffer[:batch_size]
                meta = meta_buffer[:batch_size]
                patch_buffer = patch_buffer[batch_size:]
                meta_buffer = meta_buffer[batch_size:]
                
                results = self.classifier.predict_batch(batch)
                
                for j, (label, score, _) in enumerate(results):
                    if score > sensitivity:
                        s_idx, px, py = meta[j]
                        # Generate heatmap lazily (only for positives)
                        _, _, heatmap = self.classifier.predict(batch[j])
                        
                        raw_detections_by_slice[s_idx].append({
                            'slice': s_idx,
                            'x': px,
                            'y': py,
                            'label': label,
                            'score': score,
                            'heatmap': heatmap
                        })

        # Process remaining items in buffer
        if patch_buffer:
            results = self.classifier.predict_batch(patch_buffer)
            for j, (label, score, _) in enumerate(results):
                if score > sensitivity:
                    s_idx, px, py = meta_buffer[j]
                    _, _, heatmap = self.classifier.predict(patch_buffer[j])
                    
                    raw_detections_by_slice[s_idx].append({
                        'slice': s_idx,
                        'x': px,
                        'y': py,
                        'label': label,
                        'score': score,
                        'heatmap': heatmap
                    })
                    
        # Apply NMS per slice and aggregate
        for i in range(len(slices)):
            dets = raw_detections_by_slice[i]
            if dets:
                dets = self.non_max_suppression(dets)
                for d in dets:
                    d['instance_number'] = slices[i].InstanceNumber
                    all_detections.append(d)
                    
        return all_detections, slices
