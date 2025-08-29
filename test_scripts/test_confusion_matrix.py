import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive environments
import matplotlib
matplotlib.use('Agg')
plt.style.use('default')
sns.set_style("whitegrid")

def create_confusion_matrix(labels, predictions, class_names, save_path):
    """Create and save confusion matrix"""
    print("Creating confusion matrix...")
    
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True, annot_kws={'size': 8})
    
    plt.title('Confusion Matrix - Hybrid YOLOv9 + EfficientNet-B3 Model', 
             fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")
    return cm

def main():
    """Test confusion matrix generation with mock data"""
    print("=== Testing Confusion Matrix Generation ===")
    
    # Mock class names (same as your dataset)
    class_names = ['aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust', 
                   'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf', 
                   'spetoria', 'tan_spot', 'yellow_rust']
    
    # Generate mock predictions and labels
    np.random.seed(42)  # For reproducible results
    n_samples = 1000
    n_classes = len(class_names)
    
    # Generate random labels
    true_labels = np.random.randint(0, n_classes, n_samples)
    
    # Generate predictions with some accuracy (85% accuracy)
    predictions = true_labels.copy()
    # Introduce some errors
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    for idx in error_indices:
        predictions[idx] = np.random.randint(0, n_classes)
    
    # Create save directory
    save_dir = '../saved_models_and_data/evaluation_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate confusion matrix
    cm_path = os.path.join(save_dir, 'test_confusion_matrix.png')
    cm = create_confusion_matrix(true_labels, predictions, class_names, cm_path)
    
    # Print some statistics
    print(f"\nConfusion Matrix Shape: {cm.shape}")
    print(f"Total Samples: {n_samples}")
    print(f"Number of Classes: {n_classes}")
    
    # Calculate accuracy
    accuracy = np.sum(true_labels == predictions) / len(true_labels) * 100
    print(f"Mock Accuracy: {accuracy:.2f}%")
    
    # Show some confusion matrix values
    print(f"\nSample confusion matrix values:")
    print(f"True aphid, Predicted aphid: {cm[0, 0]}")
    print(f"True healthy, Predicted healthy: {cm[6, 6]}")
    
    print(f"\nConfusion matrix test completed successfully!")
    print(f"File saved to: {cm_path}")
    
    return cm

if __name__ == "__main__":
    confusion_matrix = main()
