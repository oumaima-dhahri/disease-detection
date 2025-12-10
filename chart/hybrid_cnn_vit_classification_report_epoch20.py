#!/usr/bin/env python3
"""
Hybrid CNN-ViT Classification Report Diagram (Epoch 20)
========================================================
Creates comprehensive classification report visualization showing:
1. Per-class Precision, Recall, and F1-Score
2. Overall performance metrics
3. Support (number of samples per class)
4. Performance summary statistics
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Get the root project directory and save to 'other' folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OTHER_FOLDER = os.path.join(PROJECT_ROOT, 'other')
os.makedirs(OTHER_FOLDER, exist_ok=True)
OUTPUT_PATH = os.path.join(OTHER_FOLDER, 'hybrid_cnn_vit_classification_report_epoch20.png')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_classification_report_diagram():
    """Create classification report diagram for Hybrid CNN-ViT (Epoch 20)"""
    
    # Classification report data from epoch 20
    classes = ['Aphid', 'Army Worm', 'Black Rust', 'Brown Rust', 'Common Rust',
               'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 
               'Powdery Mildew', 'Septoria', 'Tan Spot', 'Yellow Rust']
    
    precision = [0.9744, 0.9767, 0.8333, 0.9149, 0.9630, 0.9459, 0.9459, 
                 0.8158, 0.9107, 0.9111, 0.6286, 1.0000]
    
    recall = [0.8636, 0.9767, 0.8696, 0.9773, 0.9811, 1.0000, 0.9722, 
              0.6596, 0.9444, 1.0000, 0.5946, 1.0000]
    
    f1_scores = [0.9157, 0.9767, 0.8511, 0.9451, 0.9720, 0.9722, 0.9589, 
                 0.7294, 0.9273, 0.9535, 0.6111, 1.0000]
    
    support = [44, 43, 46, 44, 53, 35, 72, 47, 54, 41, 37, 47]
    
    # Overall metrics
    accuracy = 0.9094
    macro_avg_precision = 0.9017
    macro_avg_recall = 0.9033
    macro_avg_f1 = 0.9011
    weighted_avg_precision = 0.9074
    weighted_avg_recall = 0.9094
    weighted_avg_f1 = 0.9070
    total_samples = 563
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1.2, 1, 0.8])
    
    # Main title
    fig.suptitle('Hybrid CNN-ViT Classification Report (Epoch 20)', 
                fontsize=22, fontweight='bold', y=0.98)
    
    # ========== SUBPLOT 1: Per-Class Metrics Bar Chart ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    # Sort by F1-score for better visualization
    sorted_data = sorted(zip(classes, precision, recall, f1_scores, support), 
                        key=lambda x: x[2], reverse=True)  # Sort by F1-score
    classes_sorted, prec_sorted, rec_sorted, f1_sorted, supp_sorted = zip(*sorted_data)
    
    x_pos = np.arange(len(classes_sorted))
    width = 0.25
    
    bars1 = ax1.bar(x_pos - width, [p*100 for p in prec_sorted], width, 
                   label='Precision', color='#2196F3', alpha=0.85, 
                   edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x_pos, [r*100 for r in rec_sorted], width, 
                   label='Recall', color='#FF9800', alpha=0.85, 
                   edgecolor='black', linewidth=1.2)
    bars3 = ax1.bar(x_pos + width, [f*100 for f in f1_sorted], width, 
                   label='F1-Score', color='#4CAF50', alpha=0.85, 
                   edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('Disease Classes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classes_sorted, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 105)
    ax1.set_yticks(np.arange(0, 101, 10))
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
    
    # Add average lines
    ax1.axhline(y=macro_avg_precision*100, color='blue', linestyle='--', 
               linewidth=2, alpha=0.6, label=f'Macro Avg Precision: {macro_avg_precision*100:.2f}%')
    ax1.axhline(y=macro_avg_recall*100, color='orange', linestyle='--', 
               linewidth=2, alpha=0.6, label=f'Macro Avg Recall: {macro_avg_recall*100:.2f}%')
    ax1.axhline(y=macro_avg_f1*100, color='green', linestyle='--', 
               linewidth=2, alpha=0.6, label=f'Macro Avg F1: {macro_avg_f1*100:.2f}%')
    
    # ========== SUBPLOT 2: Support (Number of Samples) ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    colors_support = ['#4CAF50' if f1 >= 0.9 else '#FF9800' if f1 >= 0.8 else '#F44336' 
                     for f1 in f1_sorted]
    
    bars_support = ax2.barh(x_pos, supp_sorted, color=colors_support, 
                          alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Disease Classes', fontsize=12, fontweight='bold')
    ax2.set_title('Support (Test Samples per Class)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(classes_sorted, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, supp) in enumerate(zip(bars_support, supp_sorted)):
        ax2.text(supp + 1, bar.get_y() + bar.get_height()/2, 
                f'{supp}', va='center', fontsize=9, fontweight='bold')
    
    # Add total samples annotation
    ax2.text(0.98, 0.02, f'Total Test Samples: {total_samples}', 
            transform=ax2.transAxes, fontsize=11, fontweight='bold',
            ha='right', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ========== SUBPLOT 3: Overall Metrics Summary ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    metrics_names = ['Accuracy', 'Macro\nPrecision', 'Macro\nRecall', 'Macro\nF1',
                     'Weighted\nPrecision', 'Weighted\nRecall', 'Weighted\nF1']
    metrics_values = [accuracy*100, macro_avg_precision*100, macro_avg_recall*100, 
                     macro_avg_f1*100, weighted_avg_precision*100, 
                     weighted_avg_recall*100, weighted_avg_f1*100]
    
    colors_metrics = ['#4CAF50' if v >= 90 else '#FF9800' if v >= 85 else '#F44336' 
                      for v in metrics_values]
    
    bars_metrics = ax3.barh(range(len(metrics_names)), metrics_values, 
                           color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax3.set_xlabel('Performance (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Metrics', fontsize=12, fontweight='bold')
    ax3.set_title('Overall Performance Summary', fontsize=14, fontweight='bold', pad=10)
    ax3.set_yticks(range(len(metrics_names)))
    ax3.set_yticklabels(metrics_names, fontsize=10)
    ax3.set_xlim(85, 92)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars_metrics, metrics_values):
        ax3.text(val + 0.15, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')
    
    # ========== SUBPLOT 4: Performance Distribution ==========
    ax4 = fig.add_subplot(gs[2, :])
    
    # Create performance categories
    excellent = sum(1 for f1 in f1_scores if f1 >= 0.95)
    good = sum(1 for f1 in f1_scores if 0.85 <= f1 < 0.95)
    fair = sum(1 for f1 in f1_scores if 0.70 <= f1 < 0.85)
    poor = sum(1 for f1 in f1_scores if f1 < 0.70)
    
    categories = ['Excellent\n(≥95%)', 'Good\n(85-95%)', 'Fair\n(70-85%)', 'Poor\n(<70%)']
    counts = [excellent, good, fair, poor]
    colors_cat = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
    
    bars_cat = ax4.bar(categories, counts, color=colors_cat, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Performance Category (F1-Score)', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Distribution by Category', fontsize=14, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and percentages
    total_classes = len(classes)
    for bar, count in zip(bars_cat, counts):
        height = bar.get_height()
        percentage = (count / total_classes) * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # ========== Add Summary Statistics Box ==========
    summary_text = f"""Classification Report Summary (Epoch 20):

Overall Accuracy: {accuracy*100:.2f}%
Macro Average F1-Score: {macro_avg_f1*100:.2f}%
Weighted Average F1-Score: {weighted_avg_f1*100:.2f}%

Total Test Samples: {total_samples}
Number of Classes: {total_classes}

Best Performing Classes:
• Yellow Rust: 100.0% F1
• Army Worm: 97.67% F1
• Common Rust: 97.20% F1

Challenging Classes:
• Tan Spot: 61.11% F1
• Leaf Blight: 72.94% F1
• Black Rust: 85.11% F1"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', 
                     alpha=0.9, pad=12), fontfamily='monospace',
            verticalalignment='bottom')
    
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Classification report diagram saved as: {OUTPUT_PATH}")
    plt.show()

if __name__ == "__main__":
    print("Creating Hybrid CNN-ViT Classification Report Diagram (Epoch 20)...")
    create_classification_report_diagram()
    print("\n✓ Diagram generation complete!")

