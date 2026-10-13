import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import io

class AgroDocPredictor:
    def __init__(self, model_path, num_classes=38):
        # RTX 2050 CUDA use karega, nahi toh CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model architecture load karo
        self.model = get_model(num_classes).to(self.device)
        
        # Trained weights (.pth file) load karo
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        # Model ko strict Evaluation mode mein daalo (dropout aur batchnorm freeze karne ke liye)
        self.model.eval()
        
        # Same normalization aur resizing jo training ke validation loop mein thi
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_bytes, class_names):
        """
        Takes raw image bytes (from FastAPI), preprocesses it, 
        and returns the predicted disease name and confidence.
        """
        # Bytes ko PIL Image mein convert karna
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Transform apply karke batch dimension add karna (1, C, H, W)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Probabilities nikalne ke liye Softmax apply karo
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score