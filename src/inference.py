import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import io

class AgroDocPredictor:
    def __init__(self, model_path, num_classes=38):
        # Set device to CUDA if available, otherwise fallback to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model architecture and move to the target device
        self.model = get_model(num_classes).to(self.device)
        
        # Load trained weights into the model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        # Set model to evaluation mode (disables dropout and batch normalization updates)
        self.model.eval()
        
        # Define image preprocessing pipeline matching validation transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_bytes, class_names):
        """
        Takes raw image bytes (from FastAPI), preprocesses it, 
        and returns the predicted disease name and confidence score.
        """
        # Convert raw bytes to a PIL RGB Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms and add batch dimension: (1, C, H, W)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Apply softmax to calculate class probabilities
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score