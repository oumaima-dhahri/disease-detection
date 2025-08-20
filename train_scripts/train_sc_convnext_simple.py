import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

# Import the existing SC-ConvNeXt model
from sc_convnext_model import SCConvNeXt, FocalLoss

# Fix matplotlib for display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = '../dataset'
SAVE_DIR = '../saved_models_and_data'
SPLIT_OUTPUT_DIR = '../dataset_split'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 15
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 7
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)

print('Setting up image transformations for training and testing...')

# -----------------------------
# Enhanced Transformations
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('Enhanced image transformations are ready.')

print('Defining custom dataset class for wheat disease images...')

# -----------------------------
# Custom Dataset
# -----------------------------
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

print('Preparing data loaders and splitting dataset if needed...')

# -----------------------------
# Load and Split Dataset
# -----------------------------
def get_data_loaders():
    split_dirs = [os.path.join(SPLIT_OUTPUT_DIR, split) for split in ['train', 'val', 'test']]
    split_exists = all(os.path.isdir(d) and len([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))]) > 0 for d in split_dirs)
    
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
        
        train_size = int((1 - TEST_SIZE - VAL_SIZE) * len(full_dataset))
        val_size = int(VAL_SIZE * len(full_dataset))
        
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
        
        train_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'train'), transform=train_transform)
        val_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'val'), transform=test_transform)
        test_dataset = WheatDiseaseDataset(os.path.join(SPLIT_OUTPUT_DIR, 'test'), transform=test_transform)
    
    print('Calculating class weights for balanced sampling...')
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print('Data loaders are ready.')
    return train_loader, val_loader, test_loader, train_dataset.classes

# -----------------------------
# Enhanced Training Function
# -----------------------------
def train_model(model, device, train_loader, val_loader, num_epochs=EPOCHS):
    print("Starting enhanced training with SC-ConvNeXt...")
    
    # Use focal loss
    criterion = FocalLoss()
    
    # Advanced optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Advanced learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    best_acc = 0.0
    no_improvement_epochs = 0
    scaler = torch.amp.GradScaler(enabled=USE_MIXED_PRECISION)
    train_log = []
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Training phase
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=USE_MIXED_PRECISION):
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
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=USE_MIXED_PRECISION):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        # Update learning rate
        scheduler.step()
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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_sc_convnext_model.pth"))
            no_improvement_epochs = 0
            print(f"  🎯 New best validation accuracy: {best_acc:.4f}")
        else:
            no_improvement_epochs += 1
        
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
            break
    
    print("Training completed successfully! 🎉")
    return model, train_log

# -----------------------------
# Model Loading Function
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_model(num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the existing SC-ConvNeXt model
    model = SCConvNeXt(num_classes=num_classes)
    
    # Try to load SimCLR weights if available
    simclr_weights = 'saved_models_and_data/simclr_convnext_tiny.pth'
    if os.path.exists(simclr_weights):
        print(f"Loading SimCLR pre-trained weights from {simclr_weights}")
        try:
            model.load_simclr_weights(simclr_weights)
        except Exception as e:
            print(f"Error loading SimCLR weights: {e}")
            print("Continuing with ImageNet pre-trained weights")
    else:
        print("No SimCLR pre-trained weights found. Using ImageNet pre-trained weights.")
    
    model = model.to(device)
    return model, device

# -----------------------------
# Enhanced Evaluation Function with Confusion Matrix
# -----------------------------
def evaluate_model(model, test_loader, device, class_names):
    print("Evaluating model on test set...")
    
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
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Generate confusion matrix with robust error handling
    print('🎯 Generating confusion matrix...')
    
    try:
        conf_matrix = confusion_matrix(y_true, y_pred)
        print(f"✅ Confusion matrix calculated successfully: {conf_matrix.shape}")
        
        # Create a large, clear figure
        plt.figure(figsize=(16, 12))
        
        # Create heatmap with better styling
        sns.heatmap(conf_matrix, 
                    annot=True,           # Show numbers
                    fmt="d",              # Integer format
                    cmap="Blues",         # Color scheme
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar_kws={"shrink": 0.8},
                    square=True)          # Make cells square
        
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
        
        # Save high-resolution image
        save_path_high = os.path.join(SAVE_DIR, 'sc_convnext_confusion_matrix_high_res.png')
        plt.savefig(save_path_high, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ High-resolution confusion matrix saved to: {save_path_high}")
        
        # Save standard resolution
        save_path = os.path.join(SAVE_DIR, 'sc_convnext_confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Standard confusion matrix saved to: {save_path}")
        
        # Try to display the confusion matrix
        print("🖥️ Attempting to display confusion matrix...")
        try:
            plt.show()
            print("✅ Confusion matrix displayed successfully!")
        except Exception as display_error:
            print(f"⚠️ Could not display confusion matrix: {display_error}")
            print("💡 Check the saved image files instead!")
        
        # Close the plot to free memory
        plt.close()
        
        # Print confusion matrix statistics
        print("\n📊 Confusion Matrix Statistics:")
        print(f"  Total Predictions: {np.sum(conf_matrix)}")
        print(f"  Correct Predictions: {np.sum(np.diag(conf_matrix))}")
        print(f"  Incorrect Predictions: {np.sum(conf_matrix) - np.sum(np.diag(conf_matrix))}")
        
        # Calculate overall accuracy
        overall_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        print(f"  Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Error generating confusion matrix: {e}")
        print("💡 Will continue with classification report...")
        conf_matrix = None
    
    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    return test_accuracy, y_true, y_pred

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print('🚀 Starting SC-ConvNeXt Training...')
    print('=' * 60)
    
    print('Loading data...')
    train_loader, val_loader, test_loader, class_labels = get_data_loaders()
    print('Data loaded. Classes:', class_labels)
    print(f'Number of classes: {len(class_labels)}')
    
    print('Initializing model...')
    try:
        model, device = load_model(len(class_labels))
        print('✅ Model initialized successfully!')
    except Exception as e:
        print(f'❌ Error initializing model: {e}')
        exit(1)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable')
    
    print('Starting training...')
    model, train_log = train_model(model, device, train_loader, val_loader)
    
    print('Training complete. Evaluating on test set...')
    
    # Load best model for evaluation
    best_model_path = os.path.join(SAVE_DIR, "best_sc_convnext_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("✅ Loaded best model for evaluation")
    else:
        print("⚠️ Best model not found, using current model")
    
    # Evaluate model
    test_acc, y_true, y_pred = evaluate_model(model, test_loader, device, class_labels)
    
    # Save final model
    final_model_path = os.path.join(SAVE_DIR, "wheat_disease_sc_convnext_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f'✅ Final model saved to {final_model_path}')
    
    # Save training log
    import json
    training_log = {
        'model_type': 'SC-ConvNeXt with Enhanced Training',
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'image_size': IMAGE_SIZE,
        'num_classes': len(class_labels),
        'class_labels': class_labels,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'test_accuracy': test_acc,
        'training_history': train_log,
        'model_parameters': total_params,
        'trainable_parameters': trainable_params
    }
    
    with open(os.path.join(SAVE_DIR, "sc_convnext_training_log.json"), 'w') as f:
        json.dump(training_log, f, indent=2)
    print('✅ Training log saved.')
    
    # Plot training curves
    epochs = [log['epoch'] for log in train_log]
    train_losses = [log['train_loss'] for log in train_log]
    val_losses = [log['val_loss'] for log in train_log]
    train_accs = [log['train_acc'] for log in train_log]
    val_accs = [log['val_acc'] for log in train_log]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
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
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save training curves
    curves_path = os.path.join(SAVE_DIR, 'sc_convnext_training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f'✅ Training curves saved to: {curves_path}')
    
    plt.show()
    
    print("🎉 SC-ConvNeXt training completed successfully!")
    print(f"🏆 Final Test Accuracy: {test_acc:.2f}%")
    print("=" * 60)
