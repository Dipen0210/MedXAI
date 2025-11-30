import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

class MedicalDeepfakeDetector:
    def __init__(self, model_path=None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Load ResNet18
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2) # Binary classification: Fake vs Real
        
        if model_path and model_path != "yolo.pt": # Avoid loading yolo weights
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded weights from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load weights from {model_path}: {e}")
                print("Using random weights for testing.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Grad-CAM hooks
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations = output
            
        # Target the last convolutional layer in ResNet18 (layer4)
        target_layer = self.model.layer4[-1].conv2
        target_layer.register_full_backward_hook(backward_hook)
        target_layer.register_forward_hook(forward_hook)

    def _generate_gradcam(self, input_shape):
        if self.gradients is None or self.activations is None:
            return np.zeros((input_shape[0], input_shape[1]), dtype=np.uint8)
            
        grads = self.gradients
        acts = self.activations
        
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze(0)
        
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (input_shape[1], input_shape[0]))
        
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        return heatmap

    def predict(self, image_or_path):
        # Load and preprocess image
        try:
            if isinstance(image_or_path, str):
                img_cv = cv2.imread(image_or_path)
                if img_cv is None:
                    raise ValueError(f"Could not read image at {image_or_path}")
            elif isinstance(image_or_path, np.ndarray):
                img_cv = image_or_path
            else:
                raise ValueError("Input must be a file path or numpy array")

            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Error loading image: {e}")
            return "Error", 0.0, None

        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        
        # Forward pass with gradient tracking for GradCAM
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        # Get prediction
        # Threshold-based classification
        # Class 0 is Fake. If Score(0) > 0.25, classify as Fake.
        # This is necessary because the model is biased towards Class 1.
        score_fake = probs[0][0].item()
        score_real = probs[0][1].item()
        
        if score_fake > 0.25:
            class_idx = 0 # Fake
            score = score_fake
        else:
            class_idx = 1 # Real
            score = score_real
            
        classes = ["Fake", "Real"]
        label = classes[class_idx]
        
        # Backward pass for GradCAM
        # We need to backward on the score of the predicted class
        output[:, class_idx].backward()
        
        heatmap = self._generate_gradcam(img_cv.shape[:2])
        
        return label, score, heatmap

    def predict_batch(self, images):
        # images: list of numpy arrays (BGR)
        if not images:
            return []
            
        batch_tensors = []
        for img in images:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            batch_tensors.append(self.transform(img_pil))
            
        input_batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_batch)
            probs = F.softmax(outputs, dim=1)
            
        results = []
        classes = ["Fake", "Real"]
        
        for i in range(len(images)):
            s0 = probs[i][0].item()
            s1 = probs[i][1].item()
            
            if s0 > 0.25:
                results.append(("Fake", s0, None))
            else:
                results.append(("Real", s1, None))
                
        return results
