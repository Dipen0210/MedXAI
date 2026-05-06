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

**Tampered Scan Detected (FM):**

<img width="1101" height="722" alt="Tampered scan detection" src="https://github.com/user-attachments/assets/e8f868eb-a9c5-48a2-be71-c2c161ed619e" />

**Authentic Nodule Detected (TM):**

<img width="1113" height="773" alt="Real cancer detection" src="https://github.com/user-attachments/assets/25d53028-0eac-456a-b483-a96d976736c2" />

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
