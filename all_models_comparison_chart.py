#!/usr/bin/env python3
"""
All Models Accuracy vs F1-Score Comparison Chart
===============================================
This script creates a comprehensive chart showing accuracy vs F1-score
comparison for all models at epoch 10 and 20
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_all_models_comparison():
    """Create comprehensive accuracy vs F1-score comparison for all models"""
    
    # Data for all models
    models = ['ConvNeXt', 'SC-ConvNeXt', 'Hybrid CNN-ViT', 'Hybrid V2', 'YOLOv9+EfficientNet', 'ProtoPNet']
    
    # 10 Epochs data
    accuracy_10 = [90.41, 88.10, 88.45, 87.21, 85.61, 56.13]
    f1_score_10 = [89.99, 87.50, 88.35, 87.22, 84.81, 53.55]
    
    # 20 Epochs data
    accuracy_20 = [91.47, 91.47, 89.70, 89.70, 86.86, 69.98]
    f1_score_20 = [90.85, 91.42, 89.53, 89.53, 86.59, 70.84]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('All Models: Accuracy vs F1-Score Comparison (10 vs 20 Epochs)', 
                fontsize=20, fontweight='bold')
    
    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Chart 1: 10 Epochs Performance
    x_pos = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, accuracy_10, width, label='Accuracy (%)', 
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x_pos + width/2, f1_score_10, width, label='F1-Score (%)', 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_title('10 Epochs Performance', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(50, 95)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Chart 2: 20 Epochs Performance
    bars3 = ax2.bar(x_pos - width/2, accuracy_20, width, label='Accuracy (%)', 
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x_pos + width/2, f1_score_20, width, label='F1-Score (%)', 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_title('20 Epochs Performance', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(50, 95)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Chart 3: Accuracy Comparison (10 vs 20)
    ax3.plot(models, accuracy_10, 'o-', color=colors[0], linewidth=3, 
             markersize=8, label='10 Epochs', alpha=0.8)
    ax3.plot(models, accuracy_20, 's-', color=colors[2], linewidth=3, 
             markersize=8, label='20 Epochs', alpha=0.8)
    
    ax3.set_title('Accuracy Comparison (10 vs 20 Epochs)', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(50, 95)
    
    # Add value labels
    for i, (acc_10, acc_20) in enumerate(zip(accuracy_10, accuracy_20)):
        ax3.text(i, acc_10 + 1, f'{acc_10:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color=colors[0])
        ax3.text(i, acc_20 + 1, f'{acc_20:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color=colors[2])
    
    # Chart 4: F1-Score Comparison (10 vs 20)
    ax4.plot(models, f1_score_10, 'o-', color=colors[1], linewidth=3, 
             markersize=8, label='10 Epochs', alpha=0.8)
    ax4.plot(models, f1_score_20, 's-', color=colors[3], linewidth=3, 
             markersize=8, label='20 Epochs', alpha=0.8)
    
    ax4.set_title('F1-Score Comparison (10 vs 20 Epochs)', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(50, 95)
    
    # Add value labels
    for i, (f1_10, f1_20) in enumerate(zip(f1_score_10, f1_score_20)):
        ax4.text(i, f1_10 + 1, f'{f1_10:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color=colors[1])
        ax4.text(i, f1_20 + 1, f'{f1_20:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, color=colors[3])
    
    plt.tight_layout()
    plt.savefig('all_models_accuracy_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed single chart
    create_detailed_comparison()

def create_detailed_comparison():
    """Create a detailed single chart with comprehensive comparison"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Data
    models = ['ConvNeXt', 'SC-ConvNeXt', 'Hybrid CNN-ViT', 'Hybrid V2', 'YOLOv9+EfficientNet', 'ProtoPNet']
    accuracy_10 = [90.41, 88.10, 88.45, 87.21, 85.61, 56.13]
    f1_score_10 = [89.99, 87.50, 88.35, 87.22, 84.81, 53.55]
    accuracy_20 = [91.47, 91.47, 89.70, 89.70, 86.86, 69.98]
    f1_score_20 = [90.85, 91.42, 89.53, 89.53, 86.59, 70.84]
    
    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create grouped bar chart
    x_pos = np.arange(len(models))
    width = 0.2
    
    bars1 = ax.bar(x_pos - 1.5*width, accuracy_10, width, label='Accuracy (10E)', 
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x_pos - 0.5*width, f1_score_10, width, label='F1-Score (10E)', 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x_pos + 0.5*width, accuracy_20, width, label='Accuracy (20E)', 
                   color=colors[2], alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax.bar(x_pos + 1.5*width, f1_score_20, width, label='F1-Score (20E)', 
                   color=colors[3], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize chart
    ax.set_title('All Models: Accuracy vs F1-Score Comparison (10 vs 20 Epochs)', 
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(50, 95)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=8)
    
    # Add performance statistics
    stats_text = """Performance Summary:
    
Best 10E Accuracy: ConvNeXt (90.41%)
Best 20E Accuracy: ConvNeXt & SC-ConvNeXt (91.47%)
Best 10E F1-Score: ConvNeXt (89.99%)
Best 20E F1-Score: SC-ConvNeXt (91.42%)

Biggest Improvement: SC-ConvNeXt (+3.37%)
Most Consistent: ConvNeXt (minimal variance)
Most Efficient: ConvNeXt (1.0h → 1.7h)"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Add benchmark lines
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, linewidth=2, label='90% Benchmark')
    ax.axhline(y=85, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='85% Baseline')
    
    plt.tight_layout()
    plt.savefig('all_models_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating All Models Accuracy vs F1-Score Comparison Charts...")
    create_all_models_comparison()
    print("Charts generated successfully!")
    print("Files created:")
    print("  - all_models_accuracy_f1_comparison.png")
    print("  - all_models_detailed_comparison.png")
