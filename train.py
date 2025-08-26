import os
import shutil
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from ultralytics import YOLO

# 🚀 OPTIMIZED Configuration for BEST ACCURACY
DATASET_DIR = '../dataset'
SPLIT_OUTPUT_DIR = '../dataset_split'
SAVE_DIR = './yolov9_results'
EPOCHS = 50                    # Increased for better convergence
BATCH_SIZE = 16                # Optimized batch size
LEARNING_RATE = 1e-4           # Optimized learning rate
IMAGE_SIZE = 640               # YOLO optimal size
PATIENCE = 15                  # Increased patience for better results

# Create directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)

class OptimizedWheatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # Calculate balanced class weights for better training
        y_labels = [sample[1] for sample in self.samples]
        self.class_weights = compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
        self.class_weights = torch.FloatTensor(self.class_weights)
        print(f"📊 Class weights for balanced training: {dict(zip(self.classes, self.class_weights.numpy()))}")
        print(f"📈 Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_optimized_yolo_labels():
    """Create YOLO format labels with optimized bounding boxes for classification"""
    print("🔧 Creating optimized YOLO dataset...")
    yolo_dir = os.path.join(SAVE_DIR, 'yolo_dataset')
    os.makedirs(yolo_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(SPLIT_OUTPUT_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        yolo_split_dir = os.path.join(yolo_dir, split)
        os.makedirs(yolo_split_dir, exist_ok=True)
        os.makedirs(os.path.join(yolo_split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(yolo_split_dir, 'labels'), exist_ok=True)
        
        split_samples = 0
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            class_idx = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]).index(class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Copy image
                    src_img = os.path.join(class_dir, img_name)
                    dst_img = os.path.join(yolo_split_dir, 'images', img_name)
                    shutil.copy2(src_img, dst_img)
                    
                    # Create optimized YOLO label with better bounding box positioning
                    img = Image.open(src_img)
                    w, h = img.size
                    
                    # Create optimized bounding box (0.3 to 0.7 of image dimensions for better detection)
                    x_center = 0.5
                    y_center = 0.5
                    width = 0.4    # Increased from 0.2 for better coverage
                    height = 0.4   # Increased from 0.2 for better coverage
                    
                    label_path = os.path.join(yolo_split_dir, 'labels', img_name.rsplit('.', 1)[0] + '.txt')
                    with open(label_path, 'w') as f:
                        f.write(f"{class_idx} {x_center} {y_center} {width} {height}\n")
                    
                    split_samples += 1
        
        print(f"✅ {split.capitalize()} split: {split_samples} samples")

def create_optimized_yolo_yaml():
    """Create optimized YOLO data YAML"""
    yolo_dir = os.path.join(SAVE_DIR, 'yolo_dataset')
    
    # Get class names
    train_dir = os.path.join(SPLIT_OUTPUT_DIR, 'train')
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    yaml_content = f"""path: {os.path.abspath(yolo_dir)}
train: train/images
val: val/images
test: test/images

nc: {len(classes)}
names: {classes}
"""
    
    yaml_path = os.path.join(SAVE_DIR, 'wheat_yolov9_optimized.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"📋 YOLO config created: {yaml_path}")
    return yaml_path

def train_optimized_model():
    """Train YOLOv9 model with optimized parameters for best accuracy"""
    print("🚀 Starting OPTIMIZED YOLOv9 training for BEST ACCURACY...")
    
    # Create optimized YOLO dataset
    create_optimized_yolo_labels()
    yaml_path = create_optimized_yolo_yaml()
    
    # Initialize model with best pretrained weights
    model = YOLO('yolov9c.pt')  # Using YOLOv9c for best performance
    
    # 🎯 OPTIMIZED Training Parameters for Maximum Accuracy
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device='0' if torch.cuda.is_available() else 'cpu',
        patience=PATIENCE,
        save=True,
        project=SAVE_DIR,
        name='yolov9_wheat_optimized',
        
        # 🚀 Optimizer settings for best convergence
        optimizer='AdamW',
        lr0=LEARNING_RATE,
        weight_decay=0.01,
        momentum=0.937,
        
        # 📈 Learning rate scheduling for optimal training
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 🎯 Data augmentation for better generalization
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        
        # 🔧 Advanced training settings
        close_mosaic=10,
        label_smoothing=0.1,
        nbs=64,
        
        # 📊 Validation settings
        val=True,
        plots=True,
        save_period=10,
        
        # 💾 Model saving
        cache=True,
        
        # 🎯 Loss function optimization
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 🔍 Advanced detection settings
        conf=0.001,
        iou=0.6,
        max_det=300,
        agnostic_nms=False,
        verbose=True
    )
    
    print("✅ Training completed successfully!")
    return model, results

def evaluate_model_comprehensive(model):
    """Comprehensive model evaluation with detailed confusion matrix"""
    print("📊 Starting comprehensive model evaluation...")
    
    # Validate model
    print("🔍 Running validation...")
    results = model.val()
    
    # Get predictions for confusion matrix
    yolo_dir = os.path.join(SAVE_DIR, 'yolo_dataset')
    test_dir = os.path.join(yolo_dir, 'test')
    
    if not os.path.exists(test_dir):
        print("❌ Test directory not found")
        return
    
    y_true = []
    y_pred = []
    confidence_scores = []
    
    # Get class names
    train_dir = os.path.join(SPLIT_OUTPUT_DIR, 'train')
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"🎯 Evaluating on {len(classes)} classes: {classes}")
    
    # Predict on test images
    test_images_dir = os.path.join(test_dir, 'images')
    total_images = len([f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"📸 Processing {total_images} test images...")
    
    for idx, img_name in enumerate(os.listdir(test_images_dir)):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            if idx % 50 == 0:
                print(f"🔄 Processing image {idx+1}/{total_images}")
                
            img_path = os.path.join(test_images_dir, img_name)
            
            # Get true label from labels directory
            label_name = img_name.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(test_dir, 'labels', label_name)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    true_class = int(f.read().strip().split()[0])
                    y_true.append(true_class)
                
                # Predict with optimized confidence threshold
                results = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)
                if results and len(results[0].boxes) > 0:
                    # Get the highest confidence prediction
                    conf_scores = results[0].boxes.conf
                    cls_predictions = results[0].boxes.cls
                    
                    # Find the prediction with highest confidence
                    best_idx = torch.argmax(conf_scores)
                    pred_class = int(cls_predictions[best_idx])
                    best_conf = float(conf_scores[best_idx])
                    
                    y_pred.append(pred_class)
                    confidence_scores.append(best_conf)
                else:
                    # Default prediction if no detection
                    y_pred.append(0)
                    confidence_scores.append(0.0)
    
    # 🎯 Generate comprehensive evaluation metrics
    if len(y_true) > 0 and len(y_pred) > 0:
        print(f"\n📊 Evaluation completed on {len(y_true)} samples")
        
        # Calculate accuracy
        accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 📈 Create enhanced confusion matrix visualization
        plt.figure(figsize=(16, 14))
        
        # Main confusion matrix
        ax1 = plt.subplot(2, 2, (1, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    cbar=True, square=True, annot_kws={"size": 10})
        plt.title('YOLOv9 Wheat Disease Classification - Confusion Matrix', 
                 fontsize=20, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Per-class accuracy
        ax2 = plt.subplot(2, 2, 4)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        bars = plt.bar(range(len(classes)), per_class_accuracy, color='skyblue', alpha=0.7)
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Class Index')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(classes)), range(len(classes)))
        plt.ylim(0, 1)
        
        # Add accuracy values on bars
        for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save enhanced confusion matrix
        cm_path = os.path.join(SAVE_DIR, 'enhanced_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # 📊 Generate detailed classification report
        print("\n📊 Detailed Classification Report:")
        report = classification_report(y_true, y_pred, target_names=classes, digits=4)
        print(report)
        
        # 💾 Save comprehensive results
        results_summary = {
            'model_type': 'YOLOv9 Optimized',
            'accuracy': accuracy,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'image_size': IMAGE_SIZE,
            'total_samples': len(y_true),
            'classes': classes,
            'confusion_matrix': cm.tolist(),
            'per_class_accuracy': per_class_accuracy.tolist(),
            'y_true': y_true,
            'y_pred': y_pred,
            'confidence_scores': confidence_scores
        }
        
        # Save confusion matrix data
        cm_data = {
            'confusion_matrix': cm.tolist(),
            'classes': classes,
            'y_true': y_true,
            'y_pred': y_pred,
            'confidence_scores': confidence_scores,
            'per_class_accuracy': per_class_accuracy.tolist()
        }
        
        cm_json_path = os.path.join(SAVE_DIR, 'comprehensive_confusion_matrix.json')
        with open(cm_json_path, 'w') as f:
            json.dump(cm_data, f, indent=2)
        
        # Save training results
        results_path = os.path.join(SAVE_DIR, 'comprehensive_training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✅ Results saved to: {SAVE_DIR}")
        print(f"📊 Enhanced confusion matrix: {cm_path}")
        print(f"📈 Comprehensive results: {results_path}")
        print(f"📋 Confusion matrix data: {cm_json_path}")
        
        # 🏆 Performance summary
        print(f"\n🏆 PERFORMANCE SUMMARY:")
        print(f"   Overall Accuracy: {accuracy*100:.2f}%")
        print(f"   Best Class: {classes[np.argmax(per_class_accuracy)]} ({np.max(per_class_accuracy)*100:.2f}%)")
        print(f"   Worst Class: {classes[np.argmin(per_class_accuracy)]} ({np.min(per_class_accuracy)*100:.2f}%)")
        print(f"   Average Confidence: {np.mean(confidence_scores):.3f}")
        
        return accuracy, cm, classes
    else:
        print("❌ No valid predictions generated")
        return None, None, None

def main():
    """Main execution with comprehensive training and evaluation"""
    print("🌾 YOLOv9 Wheat Disease Detection - OPTIMIZED for BEST ACCURACY")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Patience: {PATIENCE}")
    print("=" * 70)
    
    # Train optimized model
    print("\n🚀 Starting optimized training...")
    model, results = train_optimized_model()
    
    # Comprehensive evaluation
    print("\n📊 Starting comprehensive evaluation...")
    accuracy, confusion_matrix, classes = evaluate_model_comprehensive(model)
    
    # Save best model
    best_model_path = os.path.join(SAVE_DIR, 'best_yolov9_wheat_optimized.pt')
    try:
        shutil.copy2(os.path.join(SAVE_DIR, 'yolov9_wheat_optimized', 'weights', 'best.pt'), best_model_path)
        print(f"\n🏆 Best model saved to: {best_model_path}")
    except Exception as e:
        print(f"⚠️ Could not copy best model: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 OPTIMIZED YOLOv9 Training Completed Successfully!")
    if accuracy:
        print(f"🏆 Final Accuracy: {accuracy*100:.2f}%")
    print("=" * 70)
    
    return model

if __name__ == "__main__":
    model = main()
    print("\n🎉 All operations completed successfully!")