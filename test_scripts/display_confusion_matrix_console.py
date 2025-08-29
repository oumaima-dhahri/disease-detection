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

def display_confusion_matrix_console(labels, predictions, class_names):
    """Display confusion matrix directly in console output"""
    print("\n" + "="*80)
    print("CONFUSION MATRIX (Displayed in Console Output)")
    print("="*80)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Print class names as header
    header = f"{'True\\Pred':>12}"
    for class_name in class_names:
        header += f"{class_name:>10}"
    print(header)
    print("-" * (12 + len(class_names) * 10))
    
    # Print confusion matrix with class names
    for i, true_class in enumerate(class_names):
        row = f"{true_class:>12}"
        for j in range(len(class_names)):
            row += f"{cm[i, j]:>10}"
        print(row)
    
    print("-" * (12 + len(class_names) * 10))
    
    # Print summary statistics
    total_samples = np.sum(cm)
    correct_predictions = np.sum(np.diag(cm))
    accuracy = 100 * correct_predictions / total_samples
    
    print(f"\nSUMMARY:")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Print per-class accuracy
    print(f"\nPER-CLASS ACCURACY:")
    for i, class_name in enumerate(class_names):
        class_total = np.sum(cm[i, :])
        if class_total > 0:
            class_correct = cm[i, i]
            class_accuracy = 100 * class_correct / class_total
            print(f"  {class_name:>20}: {class_accuracy:>6.2f}% ({class_correct:>3}/{class_total:>3})")
        else:
            print(f"  {class_name:>20}: {0:>6.2f}% (  0/  0)")
    
    print("="*80 + "\n")
    
    return cm

def create_confusion_matrix_visualization(labels, predictions, class_names, save_path):
    """Create and save confusion matrix visualization"""
    print("Creating confusion matrix visualization...")
    
    cm = confusion_matrix(labels, predictions)
    
    # Create multiple heatmap styles
    heatmap_styles = [
        ('Blues', 'Blue Heatmap'),
        ('Reds', 'Red Heatmap'),
        ('Greens', 'Green Heatmap'),
        ('Purples', 'Purple Heatmap'),
        ('Oranges', 'Orange Heatmap'),
        ('viridis', 'Rainbow Heatmap'),
        ('coolwarm', 'CoolWarm Heatmap'),
        ('plasma', 'Plasma Heatmap')
    ]
    
    # Create a 2x4 subplot for all heatmap styles
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    for idx, (cmap, title) in enumerate(heatmap_styles):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, square=True, annot_kws={'size': 6}, ax=ax)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
        # Fix x-axis label alignment
        for label in ax.get_xticklabels():
            label.set_ha('right')
    
    plt.suptitle('Confusion Matrix - Multiple Heatmap Styles\nAdvanced EfficientNet-B5 Model for Wheat Disease Classification', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Also create individual heatmap files
    heatmaps_dir = os.path.join(os.path.dirname(save_path), 'heatmaps')
    os.makedirs(heatmaps_dir, exist_ok=True)
    
    for cmap, title in heatmap_styles:
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, square=True, annot_kws={'size': 8})
        plt.title(f'Confusion Matrix - {title}\nAdvanced EfficientNet-B5 Model for Wheat Disease Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save individual heatmap
        individual_path = os.path.join(heatmaps_dir, f'confusion_matrix_{cmap.lower().replace(" ", "_")}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"Confusion matrix visualization saved to: {save_path}")
    print(f"Individual heatmaps saved to: {heatmaps_dir}")
    return cm

def main():
    """Demonstrate confusion matrix display in console"""
    print("=== Confusion Matrix Console Display Demo ===")
    
    # Mock class names (same as your dataset)
    class_names = ['aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust', 
                   'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf', 
                   'spetoria', 'tan_spot', 'yellow_rust']
    
    # Generate mock predictions and labels (similar to your actual results)
    np.random.seed(42)  # For reproducible results
    n_samples = 563  # Same as your test dataset
    n_classes = len(class_names)
    
    # Generate random labels
    true_labels = np.random.randint(0, n_classes, n_samples)
    
    # Generate predictions with some accuracy (similar to your 9.95% accuracy)
    predictions = true_labels.copy()
    # Introduce errors to match your low accuracy
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.90), replace=False)
    for idx in error_indices:
        predictions[idx] = np.random.randint(0, n_classes)
    
    # Display confusion matrix in console
    cm = display_confusion_matrix_console(true_labels, predictions, class_names)
    
    # Create visualization
    save_dir = '../saved_models_and_data/evaluation_results'
    os.makedirs(save_dir, exist_ok=True)
    
    cm_path = os.path.join(save_dir, 'confusion_matrix_console_demo.png')
    create_confusion_matrix_visualization(true_labels, predictions, class_names, cm_path)
    
    print("Demo completed! The confusion matrix is now visible in the console output.")
    print("You can integrate this into your main test script by calling:")
    print("display_confusion_matrix_console(labels, predictions, class_names)")
    
    return cm

if __name__ == "__main__":
    confusion_matrix = main()

