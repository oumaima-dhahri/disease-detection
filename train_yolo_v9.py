import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image
import time
import warnings
import cv2
from torchvision import transforms, models
warnings.filterwarnings('ignore')

# Fix matplotlib and display issues in Kaggle
%matplotlib inline
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style and backend
plt.style.use('default')
sns.set_style("whitegrid")

# Force display
import sys
sys.stdout.flush()

# Test matplotlib
print("Testing matplotlib setup...")
try:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title('Test Plot')
    plt.show()
    print("Matplotlib test successful")
except Exception as e:
    print(f"Matplotlib test failed: {e}")

# Configuration - Matching other training scripts exactly
DATASET_DIR = '../dataset'
SAVE_DIR = '../saved_models_and_data'
SPLIT_OUTPUT_DIR = '../dataset_split'
IMAGE_SIZE = (224, 224)  # Standard size used by other scripts
BATCH_SIZE = 16          # Reduced batch size for hybrid model
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 10
LEARNING_RATE = 1e-4     # Standard learning rate used by other scripts
EARLY_STOPPING_PATIENCE = 5  # Standard patience used by other scripts
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
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

print('Defining custom dataset class for wheat disease images...')

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

print('Custom dataset class defined.')

class HybridYOLOv9EfficientNet(nn.Module):
    """Hybrid model combining YOLOv9 for detection and EfficientNet-B3 for classification"""
    def __init__(self, num_classes, pretrained=True):
        super(HybridYOLOv9EfficientNet, self).__init__()
        
        # YOLOv9 backbone for feature extraction and detection
        self.yolo_backbone = YOLO('yolov9c.pt')
        
        # EfficientNet-B3 for classification
        self.efficientnet = models.efficientnet_b3(pretrained=pretrained)
        
        # Modify EfficientNet classifier for our number of classes
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_features, num_classes)
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(num_classes + 512, 256),  # 512 from YOLO features + num_classes from EfficientNet
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Freeze YOLO backbone initially
        # Note: YOLO objects don't have trainable parameters in the same way
        # We'll handle this differently in the forward pass
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Get YOLO features (detection) - process each image individually
        yolo_features_list = []
        for i in range(batch_size):
            # Process single image
            single_image = x[i:i+1]  # Keep batch dimension
            yolo_result = self.yolo_backbone(single_image, verbose=False)
            
            # Extract features from single image
            if hasattr(yolo_result, 'boxes') and yolo_result.boxes is not None and len(yolo_result.boxes.conf) > 0:
                # Use detection confidence as features
                conf_scores = yolo_result.boxes.conf[:512]  # Limit to 512 features
                if len(conf_scores) < 512:
                    # Pad with zeros if needed
                    padding = torch.zeros(512 - len(conf_scores), device=conf_scores.device)
                    conf_scores = torch.cat([conf_scores, padding])
                yolo_features_list.append(conf_scores)
            else:
                # No detections, use zero features
                yolo_features_list.append(torch.zeros(512, device=x.device))
        
        # Stack all YOLO features into batch
        yolo_feat = torch.stack(yolo_features_list)  # Shape: (batch_size, 512)
        
        # Get EfficientNet classification
        efficientnet_out = self.efficientnet(x)
        
        # Combine features - ensure both have same dimensions
        combined_features = torch.cat([efficientnet_out, yolo_feat], dim=1)
        
        # Final classification
        output = self.feature_fusion(combined_features)
        
        return output
    
    def train(self, mode=True):
        """Custom train method to handle YOLO object"""
        # Set EfficientNet to training mode
        self.efficientnet.train(mode)
        # Set feature fusion layers to training mode
        self.feature_fusion.train(mode)
        # YOLO object doesn't need train mode setting
        return self
    
    def eval(self):
        """Custom eval method to handle YOLO object"""
        # Set EfficientNet to eval mode
        self.efficientnet.eval()
        # Set feature fusion layers to eval mode
        self.feature_fusion.eval()
        # YOLO object doesn't need eval mode setting
        return self

print('Preparing data loaders and splitting dataset if needed...')

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
    """Get image transformations for training and validation"""
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_classes, device):
    """Train the hybrid model"""
    print("Starting hybrid model training...")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)
    
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
            
            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            no_improvement_epochs = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_hybrid_model.pth'))
            print(f'  New best validation accuracy: {best_acc:.2f}%')
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
            'val_acc': val_acc
        })
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.1f} minutes")
    
    return train_log

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the trained model on test set with proper error handling"""
    try:
        print("Evaluating model on test set...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        test_correct = 0
        test_total = 0
        
        # Collect predictions and labels
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
                
                # Clear memory
                del inputs, outputs, predicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"Evaluation completed. Collected {len(all_predictions)} samples")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Validate data
        print(f"Data validation:")
        print(f"  Labels shape: {all_labels.shape}, dtype: {all_labels.dtype}")
        print(f"  Predictions shape: {all_predictions.shape}, dtype: {all_predictions.dtype}")
        print(f"  Labels range: {all_labels.min()} to {all_labels.max()}")
        print(f"  Predictions range: {all_predictions.min()} to {all_predictions.max()}")
        
        # Check for data issues
        if len(all_predictions) == 0 or len(all_labels) == 0:
            print("Error: Empty arrays detected!")
            return None, None, None
        
        if np.any(np.isnan(all_predictions)) or np.any(np.isnan(all_labels)):
            print("Error: NaN values detected!")
            return None, None, None
        
        # Ensure integer types
        all_labels = all_labels.astype(int)
        all_predictions = all_predictions.astype(int)
        
        test_accuracy = 100. * test_correct / test_total
        print(f'Test Accuracy: {test_accuracy:.2f}%')
        
        # Classification report
        try:
            print("\nClassification Report:")
            report = classification_report(all_labels, all_predictions, target_names=class_names, digits=4)
            print(report)
        except Exception as e:
            print(f"Error generating classification report: {e}")
        
        # Create confusion matrix with error handling
        try:
            print("Creating confusion matrix...")
            cm = confusion_matrix(all_labels, all_predictions)
            print(f"Confusion matrix created successfully: {cm.shape}")
            print("Confusion matrix values:")
            print(cm)
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            print(f"Unique values in labels: {np.unique(all_labels)}")
            print(f"Unique values in predictions: {np.unique(all_predictions)}")
            return None, None, None
        
        # Create and display confusion matrix plot
        try:
            print("Creating confusion matrix plot...")
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names,
                        cbar=True, square=True)
            
            # Customize plot
            plt.title('Confusion Matrix - Hybrid YOLOv9 + EfficientNet-B3 Model', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('True Label', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot first
            save_path = os.path.join(SAVE_DIR, 'hybrid_model_confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Confusion matrix saved to: {save_path}")
            
            # Display the plot
            plt.show()
            print("Confusion matrix displayed successfully")
            
            # Close the plot to free memory
            plt.close()
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
            import traceback
            traceback.print_exc()
            
            # Try alternative display method
            try:
                print("Trying alternative display method...")
                from IPython.display import display
                import matplotlib.pyplot as plt2
                
                fig, ax = plt2.subplots(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix (Alternative Display)')
                display(fig)
                plt2.close(fig)
                print("Alternative display successful")
                
            except Exception as e2:
                print(f"Alternative display also failed: {e2}")
        
        return test_accuracy, all_predictions, all_labels
        
    except Exception as e:
        print(f"Error in evaluate_model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main execution"""
    print('Loading data...')
    train_dataset, val_dataset, test_dataset = get_data_loaders()
    class_labels = train_dataset.classes
    NUM_CLASSES = len(class_labels)
    print('Data loaded. Classes:', class_labels)
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Apply transforms to datasets
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    test_dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print('Initializing hybrid model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HybridYOLOv9EfficientNet(NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    print('Model initialized. Starting training...')
    train_log = train_model(model, train_loader, val_loader, NUM_CLASSES, device)
    
    print('Training complete. Evaluating on test set...')
    test_acc, predictions, labels = evaluate_model(model, test_loader, device, class_labels)
    
    # Save final model
    final_model_path = os.path.join(SAVE_DIR, 'hybrid_yolov9_efficientnet_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved to {final_model_path}')
    
    # Save training log
    import json
    training_log = {
        'model_type': 'Hybrid YOLOv9 + EfficientNet-B3',
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
    
    with open(os.path.join(SAVE_DIR, "hybrid_model_training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=2)
    print('Training log saved.')
    
    # Plot training curves
    epochs = [log['epoch'] for log in train_log]
    train_losses = [log['train_loss'] for log in train_log]
    val_losses = [log['val_loss'] for log in train_log]
    train_accs = [log['train_acc'] for log in train_log]
    val_accs = [log['val_acc'] for log in train_log]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'hybrid_model_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Hybrid model training completed successfully!")
    return model

if __name__ == "__main__":
    model = main()
