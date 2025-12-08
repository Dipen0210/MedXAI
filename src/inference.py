import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from torchvision.models.video import r3d_18
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
        
        self.is_3d = False
        
        # Default to ResNet18 (2D) initially
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # Check if model is 3D based on loading
        if model_path and model_path != "yolo.pt":
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Check for 3D specific keys or single class output
                is_3d_weights = False
                output_classes = 2 # Default
                
                # Unwrap state dict if needed
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                    
                # Check architecture
                if 'stem.0.weight' in state_dict: # r3d_18 specific
                    is_3d_weights = True
                    self.is_3d = True
                    print("Detected 3D ResNet weights.")
                    
                # Check output classes
                if 'fc.weight' in state_dict:
                    output_classes = state_dict['fc.weight'].shape[0]
                    print(f"Model has {output_classes} output classes.")
                
                if self.is_3d:
                    # Re-init as 3D Model
                    self.model = r3d_18(pretrained=False)
                    self.model.fc = nn.Linear(self.model.fc.in_features, output_classes)
                
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
        
        if self.is_3d:
            # 3D model (r3d_18) - NO transform needed, we normalize in scanner
            self.transform_3d = None
        else:
            # 3D Transform (Video) - Normalization matches 2D usually for transfer learning
            # But we need to handle the depth dimension manually in predict
            self.transform_3d = transforms.Compose([
                transforms.Resize((112, 112)), # r3d_18 often uses 112x112 for video
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]) # Kinetics-400 means
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
            
        # Target the last convolutional layer
        if self.is_3d:
            # For r3d_18: layer4[-1].conv2 is the last conv before avgpool
            # But in r3d_18 it consists of Conv3d blocks.
            # layer4[1].conv2[0] might be the one. 
            # Let's target layer4[-1] output which is easier.
            target_layer = self.model.layer4[-1]
        else:
            target_layer = self.model.layer4[-1].conv2
            
        target_layer.register_full_backward_hook(backward_hook)
        target_layer.register_forward_hook(forward_hook)

    def _generate_gradcam(self, input_shape):
        if self.gradients is None or self.activations is None:
            return None
            
        grads = self.gradients
        acts = self.activations
        
        # Pool the gradients across channels
        weights = torch.mean(grads, dim=(2, 3, 4) if self.is_3d else (2, 3), keepdim=True)
        
        # Weighted combination of activations
        if self.is_3d:
             # acts: [B, C, D, H, W] -> cam: [B, D, H, W]
            cam = torch.sum(weights * acts, dim=1).squeeze(0)
            # Take middle slice for visualization
            mid_z = cam.shape[0] // 2
            cam = cam[mid_z]
        else:
            cam = torch.sum(weights * acts, dim=1).squeeze(0)
        
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (input_shape[1], input_shape[0]))
        
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        return heatmap

    def predict(self, input_data):
        """
        Generic predict for both 2D (image path/array) and 3D (volume array).
        input_data: 
            - str: path to image (2D)
            - np.ndarray: image (2D) or volume (3D) [Depth, H, W] or [H, W]
        """
        img_cv = None
        input_tensor = None
        
        try:
            if self.is_3d:
                # 3D Case: input_data must be (Depth, H, W) numpy array
                if not isinstance(input_data, np.ndarray) or input_data.ndim != 3:
                     # Fallback if allowed? No, 3D model needs 3D input.
                     # If we get 2D, we might duplicate it?
                     if isinstance(input_data, str):
                        # Read single image and replicate?
                        img_2d = cv2.imread(input_data, cv2.IMREAD_GRAYSCALE)
                        input_data = np.stack([img_2d]*16) # Replicate to 16 depth
                     elif input_data.ndim == 2:
                        input_data = np.stack([input_data]*16)
                
                # Preprocess 3D
                # Expected: (C, D, H, W). We have (D, H, W).
                # Convert to Float tensor [0-1]
                vol = torch.from_numpy(input_data).float() / 255.0 
                # Resize D to 16 if needed? No, let's assume input is 16.
                # Resize H, W to 112
                # shape: [D, H, W] -> [1, D, H, W] (C=1) -> replicate to C=3
                vol = vol.unsqueeze(0).repeat(3, 1, 1, 1) # [3, D, H, W]
                
                # Resize spatial dims only
                # Iterate over depth to apply 2D resize? Or use interpolate?
                # Transform expect PIL usually.
                # Let's use F.interpolate
                vol = F.interpolate(vol.unsqueeze(0), size=(input_data.shape[0], 112, 112), mode='trilinear', align_corners=False).squeeze(0)
                
                # Normalize (manual standard)
                mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
                std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
                vol = (vol - mean) / std
                
                input_tensor = vol.unsqueeze(0).to(self.device) # [1, 3, D, H, W]
                img_cv = input_data[input_data.shape[0]//2] # Middle slice for heatmap context
                
            else:
                # 2D Case
                if isinstance(input_data, str):
                    img_cv = cv2.imread(input_data)
                elif isinstance(input_data, np.ndarray):
                    img_cv = input_data
                
                img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        except Exception as e:
            print(f"Error preparing input: {e}")
            return "Error", 0.0, None

        # Forward
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Scoring
        if output.shape[1] == 1:
            # Binary Sigmoid
            score_fake = torch.sigmoid(output).item()
            score_real = 1.0 - score_fake
            
            if score_fake > 0.5:
                label = "Fake"
                score = score_fake
            else:
                label = "Real"
                score = score_real
                
            # GradCAM Backprop
            output.backward()
            
        else:
            # Binary Softmax
            probs = F.softmax(output, dim=1)
            score_fake = probs[0][0].item()
            score_real = probs[0][1].item()
            
            if score_fake > 0.25: # Sensitivity threshold
                class_idx = 0
                score = score_fake
            else:
                class_idx = 1
                score = score_real
            
            classes = ["Fake", "Real"]
            label = classes[class_idx]
            output[:, class_idx].backward()

        # Generate heatmap (skip for 3D for now - causes OpenCV errors)
        if self.is_3d:
            heatmap = None  # Skip Grad-CAM for 3D
        else:
            heatmap = self._generate_gradcam(img_cv.shape[:2])
        
        return label, score, heatmap

    def predict_batch(self, inputs):
        """
        inputs: list of 
            - 2D: numpy arrays (BGR)
            - 3D: numpy arrays (D, H, W)
        """
        if not inputs: return []
        
        try:
            if self.is_3d:
                batch_tensors = []
                for vol_np in inputs:
                     # Preprocess 3D
                    # volume: numpy array (D, H, W) - ALREADY normalized by scanner to [0,1]
                    if isinstance(vol_np, np.ndarray):
                        # Just convert to tensor (D, H, W) -> (C=3, D, H, W) 
                        # by replicating across channels
                        volume_tensor = torch.from_numpy(vol_np).float()  # (D, H, W)
                        volume_tensor = volume_tensor.unsqueeze(0).repeat(3, 1, 1, 1)  # (3, D, H, W)
                        batch_tensors.append(volume_tensor)
                    else:
                        raise ValueError("3D model expects numpy array volumes.")
                    
                input_batch = torch.stack(batch_tensors).to(self.device)
            else:
                batch_tensors = []
                for img in inputs:
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    batch_tensors.append(self.transform(img_pil))
                input_batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_batch)
                
            results = []
            
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs)
                for i in range(len(inputs)):
                    s_fake = probs[i].item()
                    if s_fake > 0.5:
                        results.append(("Fake", s_fake, None))
                    else:
                        results.append(("Real", 1.0 - s_fake, None))
            else:
                probs = F.softmax(outputs, dim=1)
                for i in range(len(inputs)):
                    s0 = probs[i][0].item()
                    s1 = probs[i][1].item()
                    if s0 > 0.25:
                        results.append(("Fake", s0, None))
                    else:
                        results.append(("Real", s1, None))
                        
            return results
            
        except Exception as e:
            print(f"Batch prediction error: {e}")
            return []

