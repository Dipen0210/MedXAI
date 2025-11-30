import streamlit as st
import os
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
mode = st.sidebar.radio("Mode", ["Classify Cropped Patch", "Scan Full Slice (2D)", "Full 3D Scan (DICOM)"])
model_path = st.sidebar.text_input("Model Path", "model.pth")

def load_detector(path):
    if not os.path.exists(path): return None
    return MedicalDeepfakeDetector(path)

def load_scanner(path):
    if not os.path.exists(path): return None
    return FullScanDetector(path)

# --- SINGLE PATCH MODE ---
if mode == "Classify Cropped Patch":
    st.subheader("Classify Cropped Patch")
    st.info("ℹ️ **Note:** This mode analyzes the *entire* uploaded image. Use this for small, pre-cropped patches. To find tumors in a full CT slice, use **Scan Full Slice**.")
    st.write("Upload a small image patch (e.g. 64x64) to classify it.")
    
    detector = load_detector(model_path)
    if not detector:
        st.error("Model not found!")
        st.stop()
        
    uploaded_file = st.file_uploader("Choose a Patch...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Patch", width="stretch")
            
        with col2:
            with st.spinner('Analyzing...'):
                label, score, heatmap = detector.predict(tfile.name)
                
                # Overlay
                opencv_image = np.array(image) 
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
                opencv_image = cv2.resize(opencv_image, (64, 64))
                heatmap = cv2.resize(heatmap, (opencv_image.shape[1], opencv_image.shape[0]))
                superimposed = heatmap * 0.4 + opencv_image
                superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
                superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
                
                if label == "Real":
                    st.success(f"### Result: {label}")
                else:
                    st.error(f"### Result: {label} (Tampered)")
                st.write(f"Confidence: {score*100:.2f}%")
                st.image(superimposed, caption="Grad-CAM Explanation", width="stretch")
                
                # LLM Explanation (Single Patch)
                HF_TOKEN = os.environ.get("HF_TOKEN")
                explainer = LLMExplainer(HF_TOKEN)
                
                if st.button(f"🤖 Ask AI Doctor"):
                    with st.spinner("Consulting Llama-3..."):
                        explanation = explainer.explain(label, score, "Single Patch")
                        st.info(f"**Dr. Llama:** {explanation}")

# --- 2D SLICE SCANNER ---
elif mode == "Scan Full Slice (2D)":
    st.subheader("Scan Full Slice")
    st.write("Upload a full CT slice (PNG/JPG) to scan for hidden cancer/tampering.")
    
    scanner = load_scanner(model_path)
    if not scanner:
        st.error("Model not found!")
        st.stop()
        
    uploaded_file = st.file_uploader("Choose a Slice...", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('L') # Convert to grayscale
        # Resize to standard CT size (512x512) for consistent patch analysis
        image = image.resize((512, 512))
        img_array = np.array(image)
        
        # Normalize to 0-255 to ensure contrast for detection
        # This fixes issues where dark images (max < 80) were being ignored
        img_array = img_array.astype(float)
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-5)
        img_array = (img_array * 255).astype(np.uint8)
        
        # HARD MASKING (In App): Physically remove table and edges
        # We do this here so you can SEE it in the UI.
        h, w = img_array.shape
        img_array[int(h*0.80):, :] = 0  # Black out bottom 20%
        img_array[:, :int(w*0.05)] = 0  # Black out left 5%
        img_array[:, int(w*0.95):] = 0  # Black out right 5%
            
        st.image(image, caption="Uploaded Slice", width="stretch")
        
        if st.button("Scan Slice"):
            with st.spinner("Scanning Image..."):
                detections = scanner.scan_slice(img_array)
                st.session_state['scan_2d_results'] = detections
                
        if 'scan_2d_results' in st.session_state:
            detections = st.session_state['scan_2d_results']
            
            # Draw boxes
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            for d in detections:
                x, y = d['x'], d['y']
                color = (0, 255, 0) if d['label'] == 'Real' else (0, 0, 255)
                cv2.rectangle(img_bgr, (x-32, y-32), (x+32, y+32), color, 2)
                cv2.putText(img_bgr, f"{d['label']} {d['score']:.2f}", (x-30, y-35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if detections:
                st.warning(f"Found {len(detections)} detections!")
                
                # Layout matching 3D Report
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(img_bgr, caption="Scan Results", width="stretch")
                
                with col2:
                    for i, d in enumerate(detections):
                        st.write(f"**Analysis #{i+1}:** {d['label']} ({d['score']*100:.1f}%)")
                        st.image(d['heatmap'], caption="Grad-CAM", width="stretch")
                        
                        # LLM Explanation
                        HF_TOKEN = os.environ.get("HF_TOKEN")
                        explainer = LLMExplainer(HF_TOKEN)
                        
                        if st.button(f"🤖 Ask AI Doctor", key=f"llm_2d_{i}"):
                            with st.spinner("Consulting Llama-3..."):
                                explanation = explainer.explain(d['label'], d['score'], "2D Image")
                                st.info(f"**Dr. Llama:** {explanation}")
                        st.divider()
            else:
                st.info("No suspicious regions found.")
                st.image(img_bgr, caption="Scan Results", width="stretch")

# --- FULL 3D SCAN MODE ---
else:
    st.subheader("Full 3D Scan Detection (DICOM)")
    st.write("Upload a set of DICOM files to find hidden tampering in 3D.")
    
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
            positive_slices = sorted(list(set(d['slice'] for d in detections)))
            if positive_slices:
                st.success(f"🚨 **Detections found on slices:** {positive_slices}")
                
                with st.expander("ℹ️ How to read the AI Explanation (Grad-CAM)"):
                    st.markdown("""
                    **What is this Heatmap?**
                    The AI looks at the image in "layers", just like a human brain.
                    *   **Early Layers** see simple lines and edges.
                    *   **Middle Layers** see shapes (circles, blobs).
                    *   **Final Layer** (which we show here) sees **concepts** like "Tumor" or "Tampering".
                    
                    **Color Guide:**
                    *   🔴 **Red/Yellow:** The AI is staring at this spot. It thinks this is the **Evidence**.
                    *   🔵 **Blue:** The AI is ignoring this part.
                    """)

                st.subheader("🔍 Detection Report")
                for slice_num in positive_slices:
                    # Get detections for this slice
                    slice_dets = [d for d in detections if d['slice'] == slice_num]
                    
                    # Prepare Image
                    ds = slices[slice_num]
                    img = scanner.normalize_slice(ds)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    # Draw boxes
                    for d in slice_dets:
                        x, y = d['x'], d['y']
                        color = (0, 255, 0) if d['label'] == 'Real' else (0, 0, 255)
                        cv2.rectangle(img_bgr, (x-32, y-32), (x+32, y+32), color, 2)
                        cv2.putText(img_bgr, f"{d['label']} {d['score']:.2f}", (x-30, y-35), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Display
                    st.markdown(f"### Slice {ds.InstanceNumber}")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.image(img_bgr, width="stretch", clamp=True)
                    with col2:
                        for i, d in enumerate(slice_dets):
                            st.write(f"**Analysis #{i+1}:** {d['label']} ({d['score']*100:.1f}%)")
                            st.image(d['heatmap'], caption="Grad-CAM", width="stretch")
                            
                            # LLM Explanation
                            if st.button(f"🤖 Ask AI Doctor (Analysis #{i+1})", key=f"llm_{slice_num}_{i}"):
                                with st.spinner("Consulting Llama-3..."):
                                    explanation = explainer.explain(d['label'], d['score'], ds.InstanceNumber)
                                    st.info(f"**Dr. Llama:** {explanation}")
                    st.divider()
            else:
                st.info("✅ No detections found in the entire scan.")
                
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
                            st.image(d['heatmap'], caption="Grad-CAM Heatmap", width="stretch", clamp=True)
