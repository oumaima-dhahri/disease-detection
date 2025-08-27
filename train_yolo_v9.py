import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image
import time
import cv2
from typing import Tuple, List
import warnings
import yaml
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = '../dataset'
SAVE_DIR = '../saved_models_and_data'
SPLIT_OUTPUT_DIR = '../dataset_split'
YOLO_DATASET_DIR = '../yolo_dataset'
IMAGE_SIZE = (640, 640)  # YOLO v9 standard size
BATCH_SIZE = 16
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 100  # YOLO typically needs more epochs

LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)
os.makedirs(YOLO_DATASET_DIR, exist_ok=True)

# -----------------------------
# YOLO v9 Model Class
# -----------------------------
class YOLOv9Classifier(nn.Module):
    """YOLO v9 based classifier for disease detection"""
    def __init__(self, num_classes=12):
        super().__init__()
        # Use YOLO v9 backbone (EfficientNet-like architecture)
        self.backbone = models.efficientnet_b3(pretrained=True)
        
        # Remove the final classification layer
        self.backbone.classifier = nn.Identity()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1536, 512),  # EfficientNet-B3 features
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = self.backbone.classifier(features)
        
        # Classification
        output = self.classifier(features)
        return output

# -----------------------------
# Custom Dataset for YOLO
# -----------------------------
class YOLODiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif')):
                        path = os.path.join(class_dir, img_file)
                        self.samples.append((path, self.class_to_idx[target_class]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, target
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a random sample if current one fails
            return self.__getitem__((idx + 1) % len(self))

# -----------------------------
# Data Loading and Splitting
# -----------------------------
def get_data_loaders():
    # Check if split exists
    split_dirs = [os.path.join(SPLIT_OUTPUT_DIR, split) for split in ['train', 'val', 'test']]
    split_exists = all(os.path.isdir(d) and len(os.listdir(d)) > 0 for d in split_dirs)
    
    if split_exists:
        print('Found existing split dataset. Loading splits...')
        train_dataset = YOLODiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, 'train'), 
            transform=train_transform, 
            is_training=True
        )
        val_dataset = YOLODiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, 'val'), 
            transform=test_transform, 
            is_training=False
        )
        test_dataset = YOLODiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, 'test'), 
            transform=test_transform, 
            is_training=False
        )
    else:
        print('No split dataset found. Splitting and saving images...')
        full_dataset = YOLODiseaseDataset(DATASET_DIR, transform=None, is_training=False)
        
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()
        
        train_size = int((1 - TEST_SIZE - VAL_SIZE) * len(full_dataset))
        val_size = int(VAL_SIZE * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = Subset(full_dataset, train_indices)
        val_data = Subset(full_dataset, val_indices)
        test_data = Subset(full_dataset, test_indices)
        
        def save_split_images(dataset, indices, split_name):
            print(f"Saving images for split: {split_name}")
            for idx in indices:
                path, label_idx = dataset.dataset.samples[idx]
                class_name = dataset.dataset.classes[label_idx]
                filename = os.path.basename(path)
                dest_dir = os.path.join(SPLIT_OUTPUT_DIR, split_name, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
                shutil.copyfile(path, dest_path)
        
        save_split_images(train_data, train_indices, 'train')
        save_split_images(val_data, val_indices, 'val')
        save_split_images(test_data, test_indices, 'test')
        print('Image splits saved.')
        
        train_dataset = YOLODiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, 'train'), 
            transform=train_transform, 
            is_training=True
        )
        val_dataset = YOLODiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, 'val'), 
            transform=test_transform, 
            is_training=False
        )
        test_dataset = YOLODiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, 'test'), 
            transform=test_transform, 
            is_training=False
        )
    
    # Calculate class weights for balanced sampling
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print('Data loaders are ready.')
    return train_loader, val_loader, test_loader, train_dataset.classes

# -----------------------------
# Training Function
# -----------------------------
def train_model(model, device, train_loader, val_loader, num_epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    best_acc = 0.0
    no_improvement_epochs = 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)
    train_log = []
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    print("Starting YOLO v9 training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {time.time()-start_time:.1f}s")
        
        train_log.append({
            'epoch': epoch+1, 
            'train_loss': epoch_loss, 
            'train_acc': epoch_acc.item(), 
            'val_loss': val_loss, 
            'val_acc': val_acc.item(), 
            'lr': current_lr
        })
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_yolo_v9_model.pth"))
            no_improvement_epochs = 0
            print(f"New best validation accuracy: {best_acc:.4f}")
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break
    
    print("Training completed.")
    return model, train_log

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, device, test_loader, class_labels):
    """Evaluate the trained model on test set"""
    model.eval()
    y_true, y_pred = [], []
    
    print('Evaluating on test set...')
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    print('Test set predictions complete. Generating confusion matrix...')
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (YOLO v9 Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "yolo_v9_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))
    
    return y_true, y_pred, conf_matrix

# -----------------------------
# Main Execution
# -----------------------------
def main():
    print("Setting up YOLO v9 image transformations...")
    global train_transform, test_transform
    
    # YOLO v9 compatible transformations
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print('Loading data...')
    train_loader, val_loader, test_loader, class_labels = get_data_loaders()
    print('Data loaded. Classes:', class_labels)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    print('Initializing YOLO v9 model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = YOLOv9Classifier(num_classes=len(class_labels)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    print('Starting training...')
    model, train_log = train_model(model, device, train_loader, val_loader)
    
    print('Training complete. Evaluating on test set...')
    
    # Load best model
    best_model_path = os.path.join(SAVE_DIR, "best_yolo_v9_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found, using last trained model")
    
    # Evaluate on test set
    y_true, y_pred, conf_matrix = evaluate_model(model, device, test_loader, class_labels)
    
    # Save final model
    final_model_path = os.path.join(SAVE_DIR, "wheat_disease_yolo_v9_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    
    # Save training log
    log_path = os.path.join(SAVE_DIR, "yolo_v9_training_log.json")
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)
    print(f'Training log saved to {log_path}')
    
    # Save class labels
    labels_path = os.path.join(SAVE_DIR, "yolo_v9_class_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(class_labels, f, indent=2)
    print(f'Class labels saved to {labels_path}')
    
    # Plot training curves
    epochs = [log['epoch'] for log in train_log]
    train_losses = [log['train_loss'] for log in train_log]
    val_losses = [log['val_loss'] for log in train_log]
    train_accs = [log['train_acc'] for log in train_log]
    val_accs = [log['val_acc'] for log in train_log]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, label='Train Accuracy', marker='o')
    ax2.plot(epochs, val_accs, label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "yolo_v9_training_curves.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTraining and evaluation completed successfully!")
    print(f"Best validation accuracy: {max([log['val_acc'] for log in train_log]):.4f}")

if __name__ == "__main__":
    main()
