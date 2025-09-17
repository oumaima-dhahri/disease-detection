#!/usr/bin/env python3
"""
🚀 KAGGLE-OPTIMIZED SC-ConvNeXt Training Script
Optimized for Kaggle's environment with proper resource management
"""

import os
import json
import builtins
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image
import time
import warnings
import random
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Safely override print for real-time output
if 'original_print' not in globals():
    original_print = builtins.print

def flushed_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)

builtins.print = flushed_print

print("🚀 Starting SC-ConvNeXt Training Pipeline")
print("=" * 60)

# ============================================================================
# SC-ConvNeXt MODEL DEFINITIONS
# ============================================================================
class SCConvNeXt(nn.Module):
    def __init__(self, num_classes=12):
        super(SCConvNeXt, self).__init__()
        
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

print('✅ Model classes defined!')

# ============================================================================
# CONFIGURATION - KAGGLE OPTIMIZED
# ============================================================================
DATASET_DIR = '/kaggle/input/wheat-disease-dataset/dataset'  # Adjust path for Kaggle
SAVE_DIR = '/kaggle/working'  # Kaggle working directory
SPLIT_OUTPUT_DIR = '/kaggle/working/dataset_split'
IMAGE_SIZE = (224, 224)  # Smaller for Kaggle
BATCH_SIZE = 16  # Optimized for Kaggle GPU
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 20  # Full training epochs
LEARNING_RATE = 5e-5
EARLY_STOPPING_PATIENCE = 5
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)

print('🚀 Configuration set!')
print(f'📊 Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}')
print(f'🔧 Mixed precision: {USE_MIXED_PRECISION}')
print(f'📁 Dataset dir: {DATASET_DIR}')
print(f'💾 Save dir: {SAVE_DIR}')

# ============================================================================
# IMAGE TRANSFORMS
# ============================================================================
train_transform = transforms.Compose([
    transforms.Resize((int(IMAGE_SIZE[0] * 1.2), int(IMAGE_SIZE[1] * 1.2))),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # Reduced rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('✅ Transforms ready!')

# ============================================================================
# DATASET CLASS
# ============================================================================
class WheatDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        
        print(f"📁 Loading dataset from: {root_dir}")
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            if os.path.isdir(class_dir):
                class_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"  📂 {target_class}: {len(class_files)} images")
                for img_file in class_files:
                    path = os.path.join(class_dir, img_file)
                    self.samples.append((path, self.class_to_idx[target_class]))
        
        print(f"✅ Total samples loaded: {len(self.samples)}")
    
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
            print(f"⚠️ Error loading {path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__((idx + 1) % len(self))

print('✅ Dataset class ready!')

# ============================================================================
# DATA LOADING FUNCTION - KAGGLE OPTIMIZED
# ============================================================================
def get_data_loaders():
    print("🔄 Setting up data loaders...")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Dataset directory not found: {DATASET_DIR}")
        print("Available directories:")
        for item in os.listdir('/kaggle/input'):
            print(f"  → {item}")
        return None, None, None, None
    
    split_dirs = [os.path.join(SPLIT_OUTPUT_DIR, split) for split in ['train', 'val', 'test']]
    split_exists = all(os.path.isdir(d) and len([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]) > 0 for d in split_dirs)
    
    if split_exists:
        print('✅ Found existing split dataset. Loading splits...')
        train_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'train'), transform=train_transform)
        val_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'val'), transform=test_transform)
        test_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'test'), transform=test_transform)
    else:
        print('🔄 No split dataset found. Creating splits...')
        full_dataset = WheatDiseaseDataset(DATASET_DIR, transform=train_transform)
        
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()
        
        train_size = int((1 - TEST_SIZE - VAL_SIZE) * len(full_dataset))
        val_size = int(VAL_SIZE * len(full_dataset))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        print(f"📊 Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        train_data = Subset(full_dataset, train_indices)
        val_data = Subset(full_dataset, val_indices)
        test_data = Subset(full_dataset, test_indices)
        
        def save_split_images(dataset, indices, split_name):
            print(f"💾 Saving images for split: {split_name}")
            for i, idx in enumerate(indices):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(indices)}")
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
        print('✅ Image splits saved.')
        
        train_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'train'), transform=train_transform)
        val_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'val'), transform=test_transform)
        test_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'test'), transform=test_transform)
    
    print('🔄 Calculating class weights...')
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / np.sqrt(class_counts)
    sample_weights = [class_weights[t] for t in targets]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Kaggle optimized settings
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print('✅ Data loaders ready!')
    return train_loader, val_loader, test_loader, train_dataset.classes

# ============================================================================
# TRAINING FUNCTION - KAGGLE OPTIMIZED
# ============================================================================
def train_model(model, device, train_loader, val_loader, num_epochs=EPOCHS):
    print("🚀 Starting training with SC-ConvNeXt...")
    
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    scaler = GradScaler(enabled=USE_MIXED_PRECISION)
    
    best_acc = 0.0
    no_improvement_epochs = 0
    train_log = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        print(f"🔄 Epoch {epoch+1}/{num_epochs} - Training...")
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(enabled=USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if batch_idx % 20 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation
        print(f"🔄 Epoch {epoch+1}/{num_epochs} - Validation...")
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast(enabled=USE_MIXED_PRECISION):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"✅ Epoch {epoch+1}/{num_epochs} | "
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
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_sc_convnext_model.pth"))
            no_improvement_epochs = 0
            print(f"  🎯 New best validation accuracy: {best_acc:.4f}")
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break
    
    print("🎉 Training completed successfully!")
    return model, train_log

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_model(model, test_loader, device, class_names):
    print("🎯 Evaluating model on test set...")
    
    model.eval()
    y_true, y_pred = [], []
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if batch_idx % 20 == 0:
                print(f"Processing batch {batch_idx+1}/{len(test_loader)}")
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            test_total += labels.size(0)
            test_correct += torch.sum(preds == labels.data)
    
    test_accuracy = 100. * test_correct / test_total
    print(f'🏆 Test Accuracy: {test_accuracy:.2f}%')
    
    # Generate confusion matrix
    print('🎯 Generating confusion matrix...')
    
    try:
        conf_matrix = confusion_matrix(y_true, y_pred)
        print(f"✅ Confusion matrix calculated successfully: {conf_matrix.shape}")
        
        # Create a larger, clearer figure
        plt.figure(figsize=(14, 10))
        
        # Create heatmap with better styling
        sns.heatmap(conf_matrix, 
                    annot=True, fmt="d", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={"shrink": 0.8}, square=True)
        
        # Customize labels and title
        plt.xlabel("Predicted Label", fontsize=14, fontweight='bold')
        plt.ylabel("True Label", fontsize=14, fontweight='bold')
        plt.title("Confusion Matrix - SC-ConvNeXt Model", 
                  fontsize=16, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save confusion matrix
        save_path = os.path.join(SAVE_DIR, 'sc_convnext_confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Confusion matrix saved to: {save_path}")
        
        # Close the plot to free memory
        plt.close()
        
        # Print detailed confusion matrix statistics
        print("\n📊 Detailed Confusion Matrix Statistics:")
        print(f"  Total Predictions: {np.sum(conf_matrix)}")
        print(f"  Correct Predictions: {np.sum(np.diag(conf_matrix))}")
        print(f"  Incorrect Predictions: {np.sum(conf_matrix) - np.sum(np.diag(conf_matrix))}")
        
        # Calculate overall accuracy
        overall_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        print(f"  Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # Per-class accuracy
        print("\n📈 Per-Class Accuracy:")
        for i, class_name in enumerate(class_names):
            if np.sum(conf_matrix[i, :]) > 0:
                class_acc = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
                print(f"  {class_name:>20}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error generating confusion matrix: {e}")
        print(" Will continue with classification report...")
    
    # Generate classification report
    print("\n📋 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    return test_accuracy, y_true, y_pred

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("🚀 Starting SC-ConvNeXt Training Pipeline")
    print("=" * 60)
    
    # Set seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load data
    print('🔄 Loading data...')
    train_loader, val_loader, test_loader, class_labels = get_data_loaders()
    
    if train_loader is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    print('✅ Data loaded. Classes:', class_labels)
    print(f'📊 Number of classes: {len(class_labels)}')
    
    # Initialize model
    print('🔄 Initializing model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    model = SCConvNeXt(num_classes=len(class_labels))
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable')
    
    # Start training
    print('🚀 Starting training...')
    model, train_log = train_model(model, device, train_loader, val_loader)
    
    # Evaluate model
    print('🎯 Training complete. Evaluating...')
    
    best_model_path = os.path.join(SAVE_DIR, "best_sc_convnext_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("✅ Loaded best model for evaluation")
    else:
        print("⚠️ Best model not found, using current model")
    
    test_acc, y_true, y_pred = evaluate_model(model, test_loader, device, class_labels)
    
    # Save final model
    final_model_path = os.path.join(SAVE_DIR, "wheat_disease_sc_convnext_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f'✅ Final model saved to {final_model_path}')
    
    print("🎉 SC-ConvNeXt training completed successfully!")
    print(f"🏆 Final Test Accuracy: {test_acc:.2f}%")
    print("=" * 60)

# Execute the training pipeline
if __name__ == "__main__":
    main()
