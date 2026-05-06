# MedXAI: Explainable Deep Learning for Tampered CT Scan Detection

Deep learning pipeline to detect digitally tampered lung CT scans using the [Medical Deepfakes](https://www.kaggle.com/datasets/ymirsky/medical-deepfakes-lung-cancer) dataset. Achieves **89% accuracy** with full XAI: Grad-CAM heatmaps show where the model is looking, and Llama 3.3 70B converts that into plain-language explanations for clinicians.

## How It Works

1. **Upload** a DICOM series (full CT scan folder)
2. **3D ResNet-18** slides a 32×32×32 cube across every other slice, classifying each candidate region as tampered or authentic
3. **3D Consistency Check** discards single-slice detections — real nodules (tampered or authentic) span multiple consecutive slices
4. **Grad-CAM** highlights which voxels drove the prediction on the primary evidence slice
5. **Llama 3.3 70B** describes the heatmap finding in plain language ("The model is strongly focused on the upper-left region...")
6. **Final verdict** — FM (injected fake), TM (verified real cancer), or Healthy

## Detection Categories

| Label | Meaning |
|-------|---------|
| **FM** | False Malignant — cancer nodule digitally injected into a healthy scan |
| **TM** | True Malignant — authentic cancer nodule verified across multiple slices |
| **FB** | False Benign — real cancer digitally erased from scan |
| Healthy | No anomalies detected |

## Tech Stack

Python · PyTorch · 3D CNNs (R3D-18) · Grad-CAM · OpenCV · DICOM (pydicom) · Llama 3.3 70B · Streamlit · Explainable AI (XAI)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your Hugging Face token to a `.env` file:
```
HF_TOKEN=your_token_here
```

Run:
```bash
streamlit run app.py
```

## Output

**Authentic Nodule Detected (TM):**

<img width="1122" height="768" alt="Screenshot 2025-12-08 at 9 01 54 PM" src="https://github.com/user-attachments/assets/6404fa0d-7b56-4bd4-bc4a-95c317d2a483" />

**Tampered Scan Detected (FM):**

<img width="1101" height="722" alt="Screenshot 2025-12-08 at 9 09 05 PM" src="https://github.com/user-attachments/assets/f9b3a1ff-84ec-4ebf-b119-540ca5e55bed" />

## Project Structure

```
MedXAI/
├── app.py                  # Streamlit UI
├── src/
│   ├── inference.py        # 3D ResNet inference + Grad-CAM
│   ├── scanner.py          # Full volume scanning, NMS, 3D consistency
│   └── llm_explainer.py    # Llama 3.3 70B plain-language explanation
├── notebooks/
│   └── meddetect.ipynb     # Model training (Kaggle)
├── rsmodel.pth             # Trained model weights
└── requirements.txt
```
