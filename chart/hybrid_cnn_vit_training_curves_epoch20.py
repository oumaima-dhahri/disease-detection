#!/usr/bin/env python3
"""
Hybrid CNN-ViT Training Curves & Performance Chart (Epoch 20)
=============================================================
Creates comprehensive training visualization showing:
1. Training/Validation Loss curves over 20 epochs
2. Training/Validation Accuracy curves over 20 epochs
3. Per-class F1-Score performance
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Get the root project directory and save to 'other' folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from chart/
OTHER_FOLDER = os.path.join(PROJECT_ROOT, 'other')
os.makedirs(OTHER_FOLDER, exist_ok=True)  # Create folder if it doesn't exist
OUTPUT_PATH = os.path.join(OTHER_FOLDER, 'hybrid_cnn_vit_training_curves_epoch20.png')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_training_curves_chart():
    """Create training curves chart for Hybrid CNN-ViT (Epoch 20)"""
    
    # Training data from epoch 20 output
    epochs = np.arange(1, 21)
    
    train_loss = [1.0076, 0.4417, 0.3254, 0.2790, 0.2238, 0.1652, 0.1849, 
                  0.1475, 0.1322, 0.1346, 0.0595, 0.0602, 0.0453, 0.0390, 
                  0.0421, 0.0265, 0.0363, 0.0408, 0.0321]
    
    val_loss = [0.5277, 0.3659, 0.3901, 0.3445, 0.3099, 0.3617, 0.2797, 
                0.3835, 0.4115, 0.2889, 0.2868, 0.3039, 0.2930, 0.2866, 
                0.2875, 0.2896, 0.2897, 0.2897, 0.2897]
    
    train_acc = [0.6734, 0.8565, 0.8939, 0.9088, 0.9222, 0.9424, 0.9420, 
                 0.9500, 0.9542, 0.9554, 0.9790, 0.9813, 0.9847, 0.9874, 
                 0.9847, 0.9908, 0.9870, 0.9847, 0.9905]
    
    val_acc = [0.8185, 0.8683, 0.8737, 0.8826, 0.8950, 0.8897, 0.8915, 
               0.8950, 0.8932, 0.9217, 0.9199, 0.9181, 0.9235, 0.9270, 
               0.9253, 0.9253, 0.9253, 0.9235, 0.9235]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Hybrid CNN-ViT Training Progress (20 Epochs)', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Colors
    train_color = '#2E86AB'  # Blue
    val_color = '#A23B72'    # Purple
    
    # ========== SUBPLOT 1: Loss Curves ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, 'o-', color=train_color, linewidth=2.5, 
             markersize=6, label='Training Loss', alpha=0.8)
    ax1.plot(epochs, val_loss, 's-', color=val_color, linewidth=2.5, 
             markersize=6, label='Validation Loss', alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 21)
    ax1.set_xticks(np.arange(0, 21, 5))
    
    # Highlight best validation loss
    best_val_loss_epoch = np.argmin(val_loss) + 1
    best_val_loss = min(val_loss)
    ax1.axvline(x=best_val_loss_epoch, color='green', linestyle='--', 
                alpha=0.5, linewidth=1.5)
    ax1.text(best_val_loss_epoch, max(val_loss) * 0.9, 
            f'Best Val Loss\nEpoch {best_val_loss_epoch}\n{best_val_loss:.4f}',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', 
            facecolor='lightgreen', alpha=0.7))
    
    # ========== SUBPLOT 2: Accuracy Curves ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [acc*100 for acc in train_acc], 'o-', color=train_color, 
             linewidth=2.5, markersize=6, label='Training Accuracy', alpha=0.8)
    ax2.plot(epochs, [acc*100 for acc in val_acc], 's-', color=val_color, 
             linewidth=2.5, markersize=6, label='Validation Accuracy', alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 21)
    ax2.set_ylim(60, 100)
    ax2.set_xticks(np.arange(0, 21, 5))
    ax2.set_yticks(np.arange(60, 101, 10))
    
    # Highlight best validation accuracy
    best_val_acc_epoch = np.argmax(val_acc) + 1
    best_val_acc = max(val_acc) * 100
    ax2.axvline(x=best_val_acc_epoch, color='green', linestyle='--', 
                alpha=0.5, linewidth=1.5)
    ax2.text(best_val_acc_epoch, 75, 
            f'Best Val Acc\nEpoch {best_val_acc_epoch}\n{best_val_acc:.2f}%',
            ha='center', fontsize=9, bbox=dict(boxstyle='round', 
            facecolor='lightgreen', alpha=0.7))
    
    # ========== SUBPLOT 3: Per-Class F1-Score ==========
    ax3 = fig.add_subplot(gs[1, :])
    
    # Classification report data
    classes = ['Aphid', 'Army Worm', 'Black Rust', 'Brown Rust', 'Common Rust',
               'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 
               'Powdery Mildew', 'Septoria', 'Tan Spot', 'Yellow Rust']
    
    f1_scores = [0.9157, 0.9767, 0.8511, 0.9451, 0.9720, 0.9722, 0.9589, 
                 0.7294, 0.9273, 0.9535, 0.6111, 1.0000]
    
    precision = [0.9744, 0.9767, 0.8333, 0.9149, 0.9630, 0.9459, 0.9459, 
                 0.8158, 0.9107, 0.9111, 0.6286, 1.0000]
    
    recall = [0.8636, 0.9767, 0.8696, 0.9773, 0.9811, 1.0000, 0.9722, 
              0.6596, 0.9444, 1.0000, 0.5946, 1.0000]
    
    # Sort by F1-score for better visualization
    sorted_data = sorted(zip(classes, f1_scores, precision, recall), 
                        key=lambda x: x[1], reverse=True)
    classes_sorted, f1_sorted, prec_sorted, rec_sorted = zip(*sorted_data)
    
    x_pos = np.arange(len(classes_sorted))
    width = 0.25
    
    bars1 = ax3.bar(x_pos - width, [f*100 for f in f1_sorted], width, 
                   label='F1-Score', color='#4CAF50', alpha=0.8, 
                   edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x_pos, [p*100 for p in prec_sorted], width, 
                   label='Precision', color='#2196F3', alpha=0.8, 
                   edgecolor='black', linewidth=1)
    bars3 = ax3.bar(x_pos + width, [r*100 for r in rec_sorted], width, 
                   label='Recall', color='#FF9800', alpha=0.8, 
                   edgecolor='black', linewidth=1)
    
    ax3.set_xlabel('Disease Classes', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Class Performance Metrics (Epoch 20)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(classes_sorted, rotation=45, ha='right', fontsize=9)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 105)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only label if bar is tall enough
                ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=7, fontweight='bold')
    
    # Add average line
    avg_f1 = np.mean(f1_scores) * 100
    ax3.axhline(y=avg_f1, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Avg F1: {avg_f1:.2f}%')
    ax3.text(len(classes_sorted) - 0.5, avg_f1 + 2, 
            f'Average F1: {avg_f1:.2f}%', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ========== Add Summary Statistics ==========
    summary_text = f"""Training Summary (Epoch 20):
    
Final Training Accuracy: {train_acc[-1]*100:.2f}%
Final Validation Accuracy: {val_acc[-1]*100:.2f}%
Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_val_acc_epoch})

Final Training Loss: {train_loss[-1]:.4f}
Final Validation Loss: {val_loss[-1]:.4f}
Best Validation Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})

Overall Test Accuracy: 90.94%
Macro F1-Score: 90.11%
Weighted F1-Score: 90.70%"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightblue', 
                     alpha=0.8, pad=10), fontfamily='monospace',
            verticalalignment='bottom')
    
    plt.savefig(OUTPUT_PATH, 
               dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Chart saved as: {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    print("Creating Hybrid CNN-ViT Training Curves & Performance Chart...")
    create_training_curves_chart()
    print("\n✓ Chart generation complete!")

