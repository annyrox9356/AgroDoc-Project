import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Load a pre-trained ResNet18 model and modify the final layer 
    to match our dataset's number of classes.
    """
    # Pre-trained weights ke sath ResNet18 load karein
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Model ki aakhri layer (Fully Connected layer) ko nikal kar apni custom layer lagayein
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

# Chota sa test block
if __name__ == "__main__":
    # Test for 38 classes (PlantVillage dataset default)
    dummy_model = get_model(num_classes=38)
    print("Model Architecture ka last layer:")
    print(dummy_model.fc)
    print("\nModel successfully created for 38 classes!")