# 🫁 Medical Real Cancer Scanner

## Overview

The **Medical Real Cancer Scanner** is a deep learning-powered application designed to detect tampered or fake regions in medical imaging scans (CT scans). It utilizes a trained model to analyze images and identify potential anomalies that might indicate deepfake tampering, specifically in the context of cancer detection.

The application provides a user-friendly interface built with Streamlit, allowing users to analyze:
1.  **Cropped Patches:** Small, specific regions of an image.
2.  **Full 2D Slices:** Entire CT slices with automatic scanning.
3.  **Full 3D Scans (DICOM):** Complete volumetric scans for comprehensive analysis.

## Features

-   **Deepfake Detection:** Classifies regions as "Real" or "Tampered".
-   **Grad-CAM Visualization:** Overlays heatmaps to show *why* the model made a prediction, highlighting suspicious areas.
-   **LLM Explainer:** Integrates with a Large Language Model (Llama-3 via Hugging Face) to provide natural language explanations of the findings.
-   **Multi-Mode Analysis:**
    -   *Single Patch:* For focused analysis of pre-cropped images.
    -   *2D Slice Scanner:* Automatically scans a full image using a sliding window approach.
    -   *3D Volume Scanner:* Processes DICOM series to find anomalies across the entire volume.

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
    -   Select the mode from the sidebar (Patch, 2D, or 3D).
    -   Ensure the `Model Path` is correct (default: `model.pth`).
    -   Upload your image or DICOM files.
    -   View the results, heatmaps, and AI explanations.

## Project Structure

-   `app.py`: Main Streamlit application entry point.
-   `src/`: Source code for detection logic, scanning, and LLM integration.
    -   `inference.py`: Core model inference logic.
    -   `scanner.py`: Logic for scanning full 2D images and 3D volumes.
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
