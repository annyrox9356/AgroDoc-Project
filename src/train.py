import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import get_model
from tqdm import tqdm  # tqdm ko import karo

def train_model(num_epochs=5, batch_size=32, learning_rate=0.001):
    # 1. Device configuration - Tumhare RTX 2050 (CUDA) ko detect karega
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")

    # 2. Directories setup
    # 2. Directories setup
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Yahan 'data' ke aage folder ka naam add kiya hai
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 3. Load Data using dataset.py
    train_loader, valid_loader, class_names = get_data_loaders(DATA_DIR, batch_size=batch_size)
    num_classes = len(class_names)
    print(f"[*] Total classes to predict: {num_classes}")

    # 4. Initialize Model, Loss Function, and Optimizer
    model = get_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    # 5. The Training Loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        # ---------------- TRAINING PHASE ----------------
        model.train()  # Model ko training mode mein set karna
        running_loss = 0.0
        running_corrects = 0

        # tqdm wrapper lagane se terminal me progress bar dikhne lagega
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=True)
        for inputs, labels in train_loader:
            # Data ko GPU par bhejna
            inputs, labels = inputs.to(device), labels.to(device)

            # Purane gradients ko zero karna
            optimizer.zero_grad()

            # Forward pass: predictions nikalna
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward pass: weights update karna
            loss.backward()
            optimizer.step()

            # Loss aur accuracy track karna
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

        # ---------------- VALIDATION PHASE ----------------
        model.eval()  # Model ko evaluation mode mein set karna
        val_loss = 0.0
        val_corrects = 0

        val_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=True)
        with torch.no_grad(): # Validation ke waqt gradients calculate nahi karte (Memory bachati hai)
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(valid_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(valid_loader.dataset)
        print(f'Valid -> Loss: {val_epoch_loss:.4f} | Accuracy: {val_epoch_acc:.4f}')

        # ---------------- SAVE BEST MODEL ----------------
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            save_path = os.path.join(SAVE_DIR, 'agrodoc_best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"[*] Yay! New best model saved with Accuracy: {best_val_acc:.4f}")

    print(f"\n[*] Training Complete. Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    # Epochs ko 5 rakha hai testing ke liye. Data copy hone ke baad isko run karenge.
    train_model(num_epochs=5, batch_size=32)