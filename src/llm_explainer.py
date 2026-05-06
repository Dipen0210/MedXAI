import numpy as np
import cv2
from huggingface_hub import InferenceClient

class LLMExplainer:
    def __init__(self, api_token):
        self.client = InferenceClient(token=api_token)
        self.model_id = "meta-llama/Llama-3.3-70B-Instruct"

    def _describe_heatmap(self, heatmap):
        """Returns a plain-language description of where Grad-CAM is focused."""
        if heatmap is None:
            return None

        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY) if len(heatmap.shape) == 3 else heatmap
        h, w = gray.shape

        y, x = np.unravel_index(np.argmax(gray), gray.shape)
        intensity = np.max(gray) / 255.0

        vert = "upper" if y < h // 3 else ("middle" if y < 2 * h // 3 else "lower")
        horiz = "left" if x < w // 3 else ("center" if x < 2 * w // 3 else "right")

        strength = "strongly" if intensity > 0.7 else ("moderately" if intensity > 0.4 else "weakly")
        return f"{strength} focused on the {vert}-{horiz} portion of the scan patch"

    def explain(self, label, confidence, slice_num, heatmap=None):
        heatmap_desc = self._describe_heatmap(heatmap)

        heatmap_line = (
            f"- Grad-CAM Analysis: The model is {heatmap_desc}. "
            f"The red/yellow area in the heatmap marks this region."
            if heatmap_desc
            else "- Grad-CAM Analysis: Heatmap not available for this detection."
        )

        prompt = f"""You are a helpful medical AI assistant.
The system has analyzed a CT scan slice using a 3D deep learning model.

Detection Details:
- Diagnosis: {label} Nodule
- Confidence: {confidence*100:.1f}%
- Slice Number: {slice_num}
{heatmap_line}

Task:
Explain to a doctor what this means in 2-3 clear sentences.
Reference the Grad-CAM heatmap location to explain where the model found evidence.
If the label is "FM", explain it may be a digitally injected fake nodule.
If the label is "TM", explain it appears to be a genuine cancer nodule the model verified across multiple slices.
Keep it concise and factual."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat_completion(
                messages,
                model=self.model_id,
                max_tokens=120,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from AI: {str(e)}"
