import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    Load a pre-trained ResNet18 model and modify the final layer 
    to match our dataset's number of classes.
    """
    # Load pre-trained ResNet18 model with default weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Replace the final fully connected layer to match the target number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Test model initialization for 38 classes (PlantVillage dataset)
    dummy_model = get_model(num_classes=38)
    print("Model Architecture - Final Layer:")
    print(dummy_model.fc)
    print("\nModel successfully created for 38 classes!")