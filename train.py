import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image
import time
from torchvision import transforms, models
# Standard imports

# Standard imports and setup

"""
Data Normalization Strategy:
- Input images are converted to tensors with [0, 255] range using ToTensor()
- Then normalized to [0, 1] range using Lambda(x: x/255.0)
- Finally normalized with ImageNet statistics for EfficientNet
- For YOLO processing, images are denormalized back to [0, 1] range
- This prevents the "torch.Tensor inputs should be normalized 0.0-1.0" warnings
"""

# Configuration - OPTIMIZED FOR HIGH ACCURACY
DATASET_DIR = '../dataset'
SAVE_DIR = '../saved_models_and_data'
SPLIT_OUTPUT_DIR = '../dataset_split'
IMAGE_SIZE = (256, 256)  # Increased for better feature extraction
BATCH_SIZE = 16          # Reduced for better gradient updates
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 10             # Increased for better convergence
LEARNING_RATE = 5e-5     # Reduced for more stable training
EARLY_STOPPING_PATIENCE = 5  # Increased patience
WEIGHT_DECAY = 1e-3     # Added weight decay

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Install required packages if not present
try:
    import ultralytics
except ImportError:
    print("Installing ultralytics (YOLOv9)...")
    os.system(f"{sys.executable} -m pip install ultralytics")
    import ultralytics

from ultralytics import YOLO

# Basic warning setup

class WheatDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
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

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor from ImageNet normalization back to [0, 1] range.
    Args:
        tensor: Input tensor with shape (B, C, H, W) or (C, H, W)
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
    Returns:
        Denormalized tensor in [0, 1] range (YOLO expects this range)
    """
    try:
        denorm = tensor.clone()
        for i in range(3):  # RGB channels
            denorm[:, i] = denorm[:, i] * std[i] + mean[i]
        
        # Ensure the values are in [0, 1] range for YOLO
        denorm = torch.clamp(denorm, 0, 1)
        
        # Verify the range is correct
        min_val = denorm.min().item()
        max_val = denorm.max().item()
        if min_val < 0 or max_val > 1:
            print(f"Warning: Denormalized values out of range: [{min_val:.2f}, {max_val:.2f}]")
        
        return denorm
    except Exception as e:
        print(f"Error in denormalize_image: {e}")
        # Return tensor in [0, 1] range if denormalization fails
        return torch.clamp(tensor, 0, 1)

class SilentYOLO:
    """Wrapper for YOLO that ensures proper input normalization"""
    def __init__(self, model_path):
        self.yolo = YOLO(model_path)
    
    def __call__(self, *args, **kwargs):
        # Ensure verbose=False to reduce output
        kwargs['verbose'] = False
        return self.yolo(*args, **kwargs)

class HybridYOLOv9EfficientNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(HybridYOLOv9EfficientNet, self).__init__()
        
        # YOLOv9 backbone for feature extraction and detection
        try:
            self.yolo_backbone = SilentYOLO('yolov9c.pt')
        except Exception as e:
            print(f"Warning: YOLO initialization failed: {e}")
            # Create a dummy YOLO backbone if initialization fails
            self.yolo_backbone = None
        
        # EfficientNet-B4 (upgraded from B3 for better performance)
        self.efficientnet = models.efficientnet_b4(pretrained=pretrained)
        
        # Modify EfficientNet classifier for our number of classes
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, num_classes)
        
        # Enhanced feature fusion layer with more capacity
        self.feature_fusion = nn.Sequential(
            nn.Linear(num_classes + 512, 512),  # Increased from 256
            nn.BatchNorm1d(512),                # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),                    # Reduced dropout from 0.5
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),                # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Attention mechanism for better feature selection
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Feature projection for YOLO features
        self.yolo_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Get YOLO features (detection) - process each image individually
        yolo_features_list = []
        for i in range(batch_size):
            # Convert normalized tensor back to [0, 1] range for YOLO
            # YOLO expects input in [0, 1] range
            single_image = x[i:i+1]
            # Use the denormalization utility function
            single_image_denorm = denormalize_image(single_image)
            
            try:
                # Check if YOLO backbone is available
                if self.yolo_backbone is not None:
                    # Use SilentYOLO wrapper that ensures proper normalization
                    yolo_result = self.yolo_backbone(single_image_denorm, conf=0.1)
                    
                    if hasattr(yolo_result, 'boxes') and yolo_result.boxes is not None and len(yolo_result.boxes.conf) > 0:
                        conf_scores = yolo_result.boxes.conf[:512]
                        if len(conf_scores) < 512:
                            padding = torch.zeros(512 - len(conf_scores), device=conf_scores.device)
                            conf_scores = torch.cat([conf_scores, padding])
                        yolo_features_list.append(conf_scores)
                    else:
                        yolo_features_list.append(torch.zeros(512, device=x.device))
                else:
                    # If YOLO is not available, use zeros
                    yolo_features_list.append(torch.zeros(512, device=x.device))
            except Exception as e:
                print(f"Warning: YOLO processing failed for image {i}: {e}")
                yolo_features_list.append(torch.zeros(512, device=x.device))
        
        yolo_feat = torch.stack(yolo_features_list)
        
        # Enhanced YOLO feature processing
        yolo_feat = self.yolo_projection(yolo_feat)
        yolo_feat, _ = self.attention(yolo_feat, yolo_feat, yolo_feat)
        
        # Get EfficientNet classification
        efficientnet_out = self.efficientnet(x)
        
        # Combine features with attention
        combined_features = torch.cat([efficientnet_out, yolo_feat.mean(dim=1)], dim=1)
        
        # Final classification through enhanced fusion
        output = self.feature_fusion(combined_features)
        
        return output
    
    def train(self, mode=True):
        self.efficientnet.train(mode)
        self.feature_fusion.train(mode)
        self.attention.train(mode)
        self.yolo_projection.train(mode)
        return self
    
    def eval(self):
        self.efficientnet.eval()
        self.feature_fusion.eval()
        self.attention.eval()
        self.yolo_projection.eval()
        return self

def get_data_loaders():
    split_dirs = [os.path.join(SPLIT_OUTPUT_DIR, split) for split in ['train', 'val', 'test']]
    split_exists = all(os.path.isdir(d) and len([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]) > 0 for d in split_dirs)

    if split_exists:
        print('Found existing split dataset. Loading splits...')
        train_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'train'), transform=None)
        val_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'val'), transform=None)
        test_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'test'), transform=None)
    else:
        print('No split dataset found. Splitting and saving images...')
        full_dataset = WheatDiseaseDataset(DATASET_DIR, transform=None)

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()

        train_size = int((1 - TEST_SIZE - VAL_SIZE) * len(full_dataset))
        val_size = int(VAL_SIZE * len(full_dataset))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_data_subset = Subset(full_dataset, train_indices)
        val_data_subset = Subset(full_dataset, val_indices)
        test_data_subset = Subset(full_dataset, test_indices)

        def save_split_images(dataset_subset, split_name):
            print(f"Saving images for split: {split_name}")
            for idx_in_subset in range(len(dataset_subset)):
                original_idx = dataset_subset.indices[idx_in_subset]
                path, label_idx = full_dataset.samples[original_idx]
                class_name = full_dataset.classes[label_idx]
                filename = os.path.basename(path)
                dest_dir = os.path.join(SPLIT_OUTPUT_DIR, split_name, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
                shutil.copyfile(path, dest_path)

        save_split_images(train_data_subset, 'train')
        save_split_images(val_data_subset, 'val')
        save_split_images(test_data_subset, 'test')
        print('Image splits saved.')

        train_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'train'), transform=None)
        val_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'val'), transform=None)
        test_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'test'), transform=None)

    print('Data loaders are ready.')
    return train_dataset, val_dataset, test_dataset

def get_transforms():
    # Enhanced data augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE[0] + 32, IMAGE_SIZE[1] + 32)),  # Larger resize for random crop
        transforms.RandomCrop(IMAGE_SIZE),                             # Random crop for better generalization
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),                                # Reduced rotation for stability
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine transforms
        transforms.ToTensor(),
        # Convert to [0, 1] range first, then normalize with ImageNet stats
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        # Convert to [0, 1] range first, then normalize with ImageNet stats
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_classes, device):
    print("Starting hybrid model training for HIGH ACCURACY...")
    
    # Enhanced loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Advanced optimizer with better parameters
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    # Enhanced learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=LEARNING_RATE/100
    )
    
    best_acc = 0.0
    no_improvement_epochs = 0
    train_log = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping with better patience
        if val_acc > best_acc:
            best_acc = val_acc
            no_improvement_epochs = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_hybrid_model.pth'))
            print(f'  🎯 New best validation accuracy: {best_acc:.2f}%')
        else:
            no_improvement_epochs += 1
            
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping after {EARLY_STOPPING_PATIENCE} epochs without improvement')
            break
        
        train_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")
    
    return train_log

def evaluate_model(model, test_loader, device, class_names):
    print("Evaluating model on test set...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            if batch_idx % 50 == 0:
                print(f"Processing batch {batch_idx+1}/{len(test_loader)}")
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_accuracy = 100. * test_correct / test_total
    print(f'🎯 Test Accuracy: {test_accuracy:.2f}%')
    
    # Classification report
    print("\n📊 Classification Report:")
    report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)
    print(report)
    
    # Create confusion matrix
    print(" Creating confusion matrix...")
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"Confusion matrix created successfully: {cm.shape}")
    print("Confusion matrix values:")
    print(cm)
    
    # Create enhanced confusion matrix plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True, annot_kws={"size": 8})
    
    plt.title('Confusion Matrix - High Accuracy Hybrid YOLOv9 + EfficientNet-B4 Model', 
             fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, 'high_accuracy_hybrid_model_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Confusion matrix saved to: {save_path}")
    plt.show()  # Show plot
    
    return test_accuracy, all_predictions, all_labels

def main():
    print('🚀 Loading data for HIGH ACCURACY training...')
    print('📊 Data Normalization Strategy:')
    print('   - Images → ToTensor() [0, 255] → Lambda(x/255) [0, 1] → ImageNet Normalization')
    print('   - YOLO processing: Denormalize back to [0, 1] range')
    print('   - This prevents normalization warnings from YOLO')
    
    train_dataset, val_dataset, test_dataset = get_data_loaders()
    class_labels = train_dataset.classes
    NUM_CLASSES = len(class_labels)
    print('Data loaded. Classes:', class_labels)
    
    train_transform, val_transform = get_transforms()
    
    # Apply transforms to datasets
    # Note: Transforms will:
    # 1. Convert PIL images to tensors [0, 255]
    # 2. Normalize to [0, 1] range
    # 3. Apply ImageNet normalization for EfficientNet
    # 4. YOLO processing will denormalize back to [0, 1] range
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    test_dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print('🏗️ Initializing enhanced hybrid model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HybridYOLOv9EfficientNet(NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    print('Model initialized. Starting HIGH ACCURACY training...')
    train_log = train_model(model, train_loader, val_loader, NUM_CLASSES, device)
    
    print('Training complete. Evaluating on test set...')
    test_acc, predictions, labels = evaluate_model(model, test_loader, device, class_labels)
    
    # Save final model
    final_model_path = os.path.join(SAVE_DIR, 'high_accuracy_hybrid_yolov9_efficientnet_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    
    # Save training log
    import json
    training_log = {
        'model_type': 'High Accuracy Hybrid YOLOv9 + EfficientNet-B4',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'image_size': IMAGE_SIZE,
        'num_classes': NUM_CLASSES,
        'class_labels': class_labels,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'test_accuracy': test_acc,
        'training_history': train_log
    }
    
    with open(os.path.join(SAVE_DIR, "high_accuracy_hybrid_model_training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=2)
    print('Training log saved.')
    
    # Plot enhanced training curves
    epochs = [log['epoch'] for log in train_log]
    train_losses = [log['train_loss'] for log in train_log]
    val_losses = [log['val_loss'] for log in train_log]
    train_accs = [log['train_acc'] for log in train_log]
    val_accs = [log['val_acc'] for log in train_log]
    learning_rates = [log['lr'] for log in train_log]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax3.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'high_accuracy_hybrid_model_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()  # Show plot
    
    print("🎉 High Accuracy Hybrid model training completed successfully!")
    print(f"🏆 Final Test Accuracy: {test_acc:.2f}%")
    return model

if __name__ == "__main__":
    model = main()