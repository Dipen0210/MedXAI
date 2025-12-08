import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
from PIL import Image
import numpy as np
import cv2
import tempfile
import shutil
from src.inference import MedicalDeepfakeDetector
from src.scanner import FullScanDetector
from src.llm_explainer import LLMExplainer

# Page Config
st.set_page_config(page_title="Medical Real Cancer Detector", page_icon="🫁", layout="wide")

# Styling
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; }
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🫁 Medical Real Cancer Scanner")

# Sidebar
st.sidebar.header("Configuration")
# mode = st.sidebar.radio("Mode", ["Classify Cropped Patch", "Full 3D Scan (DICOM)"]) # Removed
model_path = st.sidebar.text_input("Model Path", "rsmodel.pth")

def load_detector(path):
    if not os.path.exists(path): return None
    return MedicalDeepfakeDetector(path)

def load_scanner(path):
    if not os.path.exists(path): return None
    return FullScanDetector(path)

# --- FULL 3D SCAN MODE ---
st.subheader("Full 3D Scan Detection (DICOM + ResNet3D)")
st.write("Upload a set of DICOM files. The system will detect digital tampering in medical scans, including:")
st.write("• **Fake injection**: Artificially added cancer nodules (FM)")
st.write("• **Cancer removal**: Digitally erased cancer regions (FB)")
st.write("• **Authentic nodules**: Real cancer detections (TM)")
st.write("*Note: Ensure the uploaded series is continuous for accurate 3D context.*")


scanner = load_scanner(model_path)
if not scanner:
    st.error("Model not found!")
    st.stop()
    
uploaded_files = st.file_uploader("Upload DICOM Series", type=["dcm"], accept_multiple_files=True)

# Initialize LLM Explainer
# In a real app, we might use st.secrets or an input field.
# Using the provided token for this session.
HF_TOKEN = os.environ.get("HF_TOKEN")
explainer = LLMExplainer(HF_TOKEN)

if uploaded_files:
    # Display Stats
    # macOS uses decimal MB (1 MB = 1,000,000 bytes). Python uses binary MiB (1 MiB = 1,048,576 bytes).
    # We switch to decimal to match the user's Finder window.
    total_size_mb = sum(f.size for f in uploaded_files) / (1000 * 1000)
    st.info(f"📂 **Upload Stats:** {len(uploaded_files)} files | Total Size: {total_size_mb:.2f} MB")
    
    temp_dir = tempfile.mkdtemp()
    for f in uploaded_files:
        with open(os.path.join(temp_dir, f.name), "wb") as out:
            out.write(f.read())
            
    if st.button("Start Scan"):
        with st.spinner("Scanning 3D Volume..."):
            # Use default sensitivity (0.85)
            detections, slices = scanner.scan(temp_dir)
            st.session_state['scan_results'] = (detections, slices)
            st.success(f"Scan Complete! Found {len(detections)} candidate regions.")
            
    if 'scan_results' in st.session_state:
        detections, slices = st.session_state['scan_results']
        
        # Show summary of positive slices
        # --- FINAL DECISION & AGGREGATION ---
        
        # --- FINAL DECISION & AGGREGATION ---
        
        # Determine Status 
        # Count FM vs TM detections
        fm_count = sum(1 for d in detections if d['label'] == 'FM')
        tm_count = sum(1 for d in detections if d['label'] == 'TM')
        
        final_status = ""
        banner_color = ""
        best_det = None
        
        if fm_count > tm_count:
            # Majority are FAKE/TAMPERED
            final_status = "TAMPERED SCAN DETECTED"
            fake_dets = [d for d in detections if d['label'] == 'FM']
            best_det = max(fake_dets, key=lambda x: x['score'])
            banner_color = "red"
            st.error(f"🚨 **Final Diagnosis: {final_status}**")
            st.warning(f"**Detected Anomalies:** FM detections: {fm_count}, TM detections: {tm_count}")
            st.warning(f"**Interpretation:** Digital tampering detected (either fake injection or cancer removal)")
            st.warning(f"**Highest anomaly confidence:** {best_det['score']:.4f}")
        elif tm_count > 0:
            # Majority or equal are REAL cancer
            final_status = "AUTHENTIC NODULE DETECTED"
            real_dets = [d for d in detections if d['label'] == 'TM']
            best_det = max(real_dets, key=lambda x: x['score'])
            banner_color = "green"
            st.success(f"✅ **Final Diagnosis: {final_status}**")
            st.info(f"**Detected Nodules:** TM detections: {tm_count}, FM detections: {fm_count}")
            st.info(f"**Verified authentic cancer nodule** with {best_det['score']*100:.1f}% confidence.")
        else:
            # No detections
            st.success("✅ **Final Diagnosis: HEALTHY / NO ANOMALIES**")
            st.write("No suspicious regions found to analyze.")
            st.stop()

        slice_num = best_det['slice']
        

        st.subheader(f"🔍 Primary Evidence ({final_status} Analysis)")
        
        # Prepare Image for the top detection
        ds = slices[slice_num]
        img = scanner.normalize_slice(ds)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 1. Composite Heatmaps for ALL detections on this slice
        slice_dets = [d for d in detections if d['slice'] == slice_num]
        
        full_heatmap = np.zeros_like(img_bgr) # Black base
        
        for d in slice_dets:
            # Skip if no heatmap (3D models don't generate them in scanner)
            if d['heatmap'] is None:
                continue
                
            # Patch geometry
            patch_h, patch_w = d['heatmap'].shape[:2]
            cx, cy = d['x'], d['y']
            x1 = cx - patch_w // 2
            y1 = cy - patch_h // 2
            x2 = x1 + patch_w
            y2 = y1 + patch_h
            
            # Boundary checks
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > img_bgr.shape[1]: x2 = img_bgr.shape[1]
            if y2 > img_bgr.shape[0]: y2 = img_bgr.shape[0]
            
            # Crop heatmap to fit
            h_x1 = max(0, -x1) if x1 < 0 else 0
            h_y1 = max(0, -y1) if y1 < 0 else 0
            h_x2 = patch_w - max(0, x2 - img_bgr.shape[1])
            h_y2 = patch_h - max(0, y2 - img_bgr.shape[0])
            
            full_heatmap[y1:y2, x1:x2] = np.maximum(
                full_heatmap[y1:y2, x1:x2],
                d['heatmap'][h_y1:h_y2, h_x1:h_x2]
            )
        
        # Overlay composite
        combined_img = cv2.addWeighted(img_bgr, 0.7, full_heatmap, 0.4, 0)
        
        # 2. Draw boxes and labels
        for d in slice_dets: # Iterate over slice_dets, not all detections
            # Color logic:
            # TM (True Malicious / Real Nodule) -> Green
            # FM (False Malicious / Fake Nodule) -> Red
            # Legacy "Fake" -> Red
            label = d['label']
            if label == 'Real' or label == 'TM':
                color = (0, 255, 0) # Green
            else:
                color = (0, 0, 255) # Red (BGR in cv2)
                
            x, y = d['x'], d['y']
            score = d['score']
            
            # Draw Box
            size = 40 # Use a fixed size for the box
            x1, y1 = x - size//2, y - size//2
            x2, y2 = x + size//2, y + size//2
            cv2.rectangle(combined_img, (x1, y1), (x2, y2), color, 2) # Draw on combined_img
            
            # Draw Label
            label_text = f"{label} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Label background
            cv2.rectangle(combined_img, (x1, y1 - h - 10), (x1 + w, y1 - 5), color, -1) # Draw on combined_img
            cv2.putText(combined_img, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1) # Draw on combined_img

        # 3. Add Decision Stamp
        stamp_color = (0, 0, 255) if final_status == "TAMPERED" else (0, 255, 0) # Red or Green
        cv2.putText(combined_img, final_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, stamp_color, 4) 
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(combined_img, caption=f"Combined Evidence: Slice {ds.InstanceNumber}", width="stretch", clamp=True)
        with col2:
            st.markdown("### 🤖 AI Doctor Analysis")
            
            # Auto-Explain (Cached in Session State)
            explanation_key = f"explain_{slice_num}_{best_det['score']}"
            if explanation_key not in st.session_state:
                with st.spinner("Dr. Llama is analyzing this case..."):
                    explanation = explainer.explain(best_det['label'], best_det['score'], ds.InstanceNumber)
                    st.session_state[explanation_key] = explanation
            
            st.info(st.session_state[explanation_key])
            
            st.divider()
            st.write(f"**Focus Area:** {best_det['label']}")
            st.write(f"**Confidence:** {best_det['score']*100:.1f}%")
        
        st.divider()
            
        st.subheader("Manual Explorer")
        slice_idx = st.slider("Scroll through Slices", 0, len(slices)-1, 0)
        
        ds = slices[slice_idx]
        img = scanner.normalize_slice(ds)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        current_detections = [d for d in detections if d['slice'] == slice_idx]
        for d in current_detections:
            x, y = d['x'], d['y']
            color = (0, 255, 0) if d['label'] == 'Real' else (0, 0, 255)
            cv2.rectangle(img_bgr, (x-32, y-32), (x+32, y+32), color, 2)
            cv2.putText(img_bgr, f"{d['label']} {d['score']:.2f}", (x-30, y-35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        st.image(img_bgr, caption=f"Slice {ds.InstanceNumber}", width="stretch", clamp=True)
        
        if current_detections:
            st.warning(f"⚠️ Found {len(current_detections)} detections on this slice!")
            for i, d in enumerate(current_detections):
                with st.expander(f"Analysis #{i+1}: {d['label']} ({d['score']*100:.1f}%)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Location:** x={d['x']}, y={d['y']}")
                        st.write(f"**Prediction:** {d['label']}")
                        st.write(f"**Confidence:** {d['score']:.4f}")
                    with col2:
                        if d['heatmap'] is not None:
                            st.image(d['heatmap'], caption="Grad-CAM Heatmap", width="stretch", clamp=True)
                        else:
                            st.info("Heatmap not available for 3D models")
