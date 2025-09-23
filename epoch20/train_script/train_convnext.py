#!/usr/bin/env python3
"""
ConvNeXt Wheat Disease Classification (Hybrid CNN-ViT Style)
Clean training and evaluation pipeline using torchvision ConvNeXt with dataset splitting
Similar structure to Hybrid CNN-ViT notebook
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import time
import shutil
import argparse
import json

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration (similar to Hybrid notebook)
# -----------------------------
DATASET_DIR = '../dataset'
SAVE_DIR = '../saved_models_and_data'
SPLIT_OUTPUT_DIR = '../dataset_split'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False

# -----------------------------
# Transformations (similar to Hybrid notebook)
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# -----------------------------
# Custom Dataset (similar to Hybrid notebook)
# -----------------------------
class WheatDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Only include subdirectories as classes
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for img_file in os.listdir(class_dir):
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
            return self.__getitem__((idx + 1) % len(self))

# -----------------------------
# ConvNeXt Model Definition (similar to Hybrid notebook structure)
# -----------------------------
class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ConvNeXt branch
        self.convnext = models.convnext_tiny(pretrained=True)
        convnext_out = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(convnext_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.convnext.features(x)  # (batch, C, H, W)
        features = features.mean(dim=[2, 3])  # Global average pool to (batch, C)
        out = self.classifier(features)
        return out

# -----------------------------
# Data Loaders (similar to Hybrid notebook)
# -----------------------------
def get_data_loaders():
    split_dirs = [os.path.join(SPLIT_OUTPUT_DIR, split) for split in ['train', 'val', 'test']]
    split_exists = all(os.path.isdir(d) and len(os.listdir(d)) > 0 for d in split_dirs)
    
    if split_exists:
        print('Found existing split dataset. Loading splits...')
        train_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'train'), transform=train_transform)
        val_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'val'), transform=test_transform)
        test_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'test'), transform=test_transform)
    else:
        print('No split dataset found. Splitting and saving images...')
        full_dataset = WheatDiseaseDataset(DATASET_DIR, transform=train_transform)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()
        
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
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
                path, label_idx = dataset.dataset.samples[idx]  # dataset is a Subset
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
        
        train_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'train'), transform=train_transform)
        val_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'val'), transform=test_transform)
        test_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'test'), transform=test_transform)
    
    print('Calculating class weights for balanced sampling...')
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print('Data loaders are ready.')
    return train_loader, val_loader, test_loader, train_dataset.classes

# -----------------------------
# Training Loop (similar to Hybrid notebook)
# -----------------------------
def train_model(model, device, train_loader, val_loader, num_epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    best_acc = 0.0
    no_improvement_epochs = 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            with torch.amp.autocast(autocast_device, enabled=USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast(autocast_device, enabled=USE_MIXED_PRECISION):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.float() / len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_convnext_model.pth"))
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    return model

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, device, test_loader, class_labels):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    print('Test set predictions complete. Generating confusion matrix...')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (ConvNeXt Model)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "convnext_confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(SAVE_DIR, 'convnext_confusion_matrix.png')}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))
    
    return y_true, y_pred, conf_matrix

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Create output directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, test_loader, class_labels = get_data_loaders()
    print(f"Number of classes: {len(class_labels)}")
    print(f"Classes: {class_labels}")
    
    # Initialize model
    model = ConvNeXtModel(num_classes=len(class_labels)).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Starting training...")
    model = train_model(model, device, train_loader, val_loader)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_convnext_model.pth"))
    print(f'Final model saved to {os.path.join(SAVE_DIR, "final_convnext_model.pth")}')
    
    # Evaluate model
    print("Starting evaluation...")
    y_true, y_pred, conf_matrix = evaluate_model(model, device, test_loader, class_labels)
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()