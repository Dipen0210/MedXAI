# 🫁 Medical Real Cancer Scanner

## Overview

The **Medical Real Cancer Scanner** is a deep learning-powered application designed to detect tampered or fake regions in medical imaging scans (CT scans). It utilizes a trained model to analyze images and identify potential anomalies that might indicate deepfake tampering, specifically in the context of cancer detection.

The application provides a user-friendly interface built with Streamlit, designed to analyze **Full 3D Scans (DICOM)** for comprehensive deepfake detection in medical imaging volumes.

## Features

-   **Deepfake Detection:** Classifies regions as "Real" or "Tampered".
-   **Grad-CAM Visualization:** Overlays heatmaps to show *why* the model made a prediction, highlighting suspicious areas.
-   **LLM Explainer:** Integrates with a Large Language Model (Llama-3 via Hugging Face) to provide natural language explanations of the findings.
-   **3D Volume Scanning:** Processes complete DICOM series to find anomalies across the entire volume with inter-slice consistency checks.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    Create a `.env` file in the root directory and add your Hugging Face token for the LLM explainer:
    ```env
    HF_TOKEN=your_hugging_face_token_here
    ```

## Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2.  **Navigate the Interface:**
    -   Upload your DICOM files.
    -   Ensure the `Model Path` is correct (default: `model.pth`).
    -   View the results, heatmaps, and AI explanations.

## Project Structure

-   `app.py`: Main Streamlit application entry point.
-   `src/`: Source code for detection logic, scanning, and LLM integration.
    -   `inference.py`: Core model inference logic.
    -   `scanner.py`: Logic for scanning 3D volumes.
    -   `llm_explainer.py`: Interface for the LLM explanation feature.
-   `model.pth`: The trained deep learning model weights.
-   `requirements.txt`: List of Python dependencies.

## Technologies

-   **Frontend:** Streamlit
-   **Deep Learning:** PyTorch, torchvision
-   **Image Processing:** OpenCV, PIL, scikit-image
-   **Medical Imaging:** pydicom
-   **LLM Integration:** Hugging Face Hub

## License

[Add License Information Here]
