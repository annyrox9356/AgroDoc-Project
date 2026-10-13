import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, img_size=(224, 224)):
    """
    Load train and validation datasets and return DataLoaders.
    """
    
    # Training Data Transformations (Data Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation Data Transformations (No Augmentation, only Resize & Normalize)
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Construct Paths
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid') # 'val' ya 'valid' jo tumhare folder ka naam ho

    # Load datasets using ImageFolder
    print(f"Loading training data from: {train_dir}")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    
    print(f"Loading validation data from: {valid_dir}")
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=val_transform)

    # Create DataLoaders
    # num_workers=2 se CPU parallel data fetch karega taaki GPU khali na baithna pade
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    class_names = train_dataset.classes
    print(f"Total Classes Found: {len(class_names)}")
    
    return train_loader, valid_loader, class_names

# Chota sa testing block: Agar is file ko direct run karoge toh ye test ho jayega
if __name__ == "__main__":
    # Ensure karo ki 'data' folder main directory mein ho
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    if os.path.exists(DATA_DIR):
        train_dl, valid_dl, classes = get_data_loaders(DATA_DIR, batch_size=16)
        print("Classes:", classes[:5], "...") # Print first 5 classes
    else:
        print(f"Dataset not found at {DATA_DIR}. Please place your 'train' and 'valid' folders inside a 'data' directory.")