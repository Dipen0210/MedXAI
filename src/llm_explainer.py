from huggingface_hub import InferenceClient

class LLMExplainer:
    def __init__(self, api_token):
        # Using the official client handles URLs automatically
        self.client = InferenceClient(token=api_token)
        self.model_id = "meta-llama/Llama-3.3-70B-Instruct"

    def explain(self, label, confidence, slice_num):
        """
        Generates a simple explanation for the detection.
        """
        prompt = f"""
        You are a helpful medical AI assistant.
        The system has detected a potential issue in a CT scan slice.
        
        Details:
        - Diagnosis: {label} Nodule
        - Confidence: {confidence*100:.1f}%
        - Slice Number: {slice_num}
        
        Task:
        Explain to the patient or doctor what this means in simple, reassuring language. 
        Mention that the "Grad-CAM heatmap" (the red area) shows where the model is looking.
        If it is "Fake", explain that it might be an injected artifact.
        If it is "Real", explain it looks like a genuine nodule.
        Keep it short (2-3 sentences).
        """
        
        try:
            # The model expects a chat format
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat_completion(
                messages, 
                model=self.model_id, 
                max_tokens=100, 
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from AI: {str(e)}"
