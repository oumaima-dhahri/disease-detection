#!/usr/bin/env python3
"""
Hybrid V2 Performance Chart with Curves
=======================================
This script creates a focused comparison chart for Hybrid V2
showing accuracy and F1-score with smooth curves between 10 vs 20 epochs
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_hybrid_v2_curves():
    """Create Hybrid V2 accuracy and F1-score comparison with smooth curves"""
    
    # Data points
    epochs = [10, 20]
    accuracy = [87.21, 89.70]
    f1_score = [87.22, 89.53]
    
    # Create smooth curves by interpolating more points
    epochs_smooth = np.linspace(10, 20, 100)
    
    # Create spline interpolation for smooth curves
    acc_spline = make_interp_spline(epochs, accuracy, k=1)
    f1_spline = make_interp_spline(epochs, f1_score, k=1)
    
    accuracy_smooth = acc_spline(epochs_smooth)
    f1_smooth = f1_spline(epochs_smooth)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Hybrid V2: Accuracy vs F1-Score Performance Curves (10 vs 20 Epochs)', 
                fontsize=18, fontweight='bold')
    
    # Colors
    colors = ['#2E86AB', '#A23B72']
    
    # Chart 1: Smooth curves comparison
    ax1.plot(epochs_smooth, accuracy_smooth, color=colors[0], linewidth=4, 
             label='Accuracy (%)', alpha=0.8)
    ax1.plot(epochs_smooth, f1_smooth, color=colors[1], linewidth=4, 
             label='F1-Score (%)', alpha=0.8)
    
    # Add data points
    ax1.scatter(epochs, accuracy, color=colors[0], s=150, zorder=5, 
               edgecolor='black', linewidth=2)
    ax1.scatter(epochs, f1_score, color=colors[1], s=150, zorder=5, 
               edgecolor='black', linewidth=2)
    
    ax1.set_title('Performance Curves Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(9, 21)
    ax1.set_ylim(86, 90.5)
    
    # Add value labels on points
    for i, (epoch, acc, f1) in enumerate(zip(epochs, accuracy, f1_score)):
        ax1.annotate(f'{acc:.2f}%', (epoch, acc), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontweight='bold', fontsize=11, color=colors[0])
        ax1.annotate(f'{f1:.2f}%', (epoch, f1), textcoords="offset points", 
                   xytext=(0,-15), ha='center', fontweight='bold', fontsize=11, color=colors[1])
    
    # Chart 2: Performance improvement trend with curves
    # Calculate improvement rates
    acc_improvement = ((89.70 - 87.21) / 87.21) * 100
    f1_improvement = ((89.53 - 87.22) / 87.22) * 100
    
    # Create improvement curves
    improvement_epochs = np.linspace(10, 20, 100)
    acc_improvement_curve = np.linspace(0, acc_improvement, 100)
    f1_improvement_curve = np.linspace(0, f1_improvement, 100)
    
    ax2.plot(improvement_epochs, acc_improvement_curve, color=colors[0], linewidth=4, 
             label=f'Accuracy Improvement (+{acc_improvement:.1f}%)', alpha=0.8)
    ax2.plot(improvement_epochs, f1_improvement_curve, color=colors[1], linewidth=4, 
             label=f'F1-Score Improvement (+{f1_improvement:.1f}%)', alpha=0.8)
    
    # Add improvement points
    ax2.scatter([10, 20], [0, acc_improvement], color=colors[0], s=150, zorder=5, 
               edgecolor='black', linewidth=2)
    ax2.scatter([10, 20], [0, f1_improvement], color=colors[1], s=150, zorder=5, 
               edgecolor='black', linewidth=2)
    
    ax2.set_title('Performance Improvement Curves', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(9, 21)
    ax2.set_ylim(-0.5, 3.5)
    
    # Add improvement annotations
    ax2.annotate(f'+{acc_improvement:.1f}%', xy=(20, acc_improvement), 
                xytext=(19, acc_improvement + 0.3),
                arrowprops=dict(arrowstyle='->', color=colors[0], lw=2),
                fontsize=12, fontweight='bold', color=colors[0], ha='center')
    
    ax2.annotate(f'+{f1_improvement:.1f}%', xy=(20, f1_improvement), 
                xytext=(19, f1_improvement - 0.3),
                arrowprops=dict(arrowstyle='->', color=colors[1], lw=2),
                fontsize=12, fontweight='bold', color=colors[1], ha='center')
    
    plt.tight_layout()
    plt.savefig('hybrid_v2_accuracy_f1_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed single chart with curves
    create_detailed_hybrid_v2_curves()

def create_detailed_hybrid_v2_curves():
    """Create a detailed single chart with smooth curves and more information"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Data points
    epochs = [10, 20]
    accuracy = [87.21, 89.70]
    f1_score = [87.22, 89.53]
    
    # Create smooth curves
    epochs_smooth = np.linspace(10, 20, 200)
    
    # Create spline interpolation for very smooth curves
    acc_spline = make_interp_spline(epochs, accuracy, k=1)
    f1_spline = make_interp_spline(epochs, f1_score, k=1)
    
    accuracy_smooth = acc_spline(epochs_smooth)
    f1_smooth = f1_spline(epochs_smooth)
    
    # Colors
    colors = ['#2E86AB', '#A23B72']
    
    # Plot smooth curves
    ax.plot(epochs_smooth, accuracy_smooth, color=colors[0], linewidth=5, 
            label='Accuracy (%)', alpha=0.9)
    ax.plot(epochs_smooth, f1_smooth, color=colors[1], linewidth=5, 
            label='F1-Score (%)', alpha=0.9)
    
    # Add data points with larger markers
    ax.scatter(epochs, accuracy, color=colors[0], s=200, zorder=5, 
              edgecolor='black', linewidth=3, alpha=0.9)
    ax.scatter(epochs, f1_score, color=colors[1], s=200, zorder=5, 
              edgecolor='black', linewidth=3, alpha=0.9)
    
    # Customize chart
    ax.set_title('Hybrid V2: Accuracy vs F1-Score Performance Curves', 
                fontweight='bold', fontsize=18, pad=25)
    ax.set_xlabel('Epochs', fontsize=16, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(9, 21)
    ax.set_ylim(86, 90.5)
    
    # Add value labels on points
    for i, (epoch, acc, f1) in enumerate(zip(epochs, accuracy, f1_score)):
        ax.annotate(f'{acc:.2f}%', (epoch, acc), textcoords="offset points", 
                   xytext=(0,15), ha='center', fontweight='bold', fontsize=13, color=colors[0])
        ax.annotate(f'{f1:.2f}%', (epoch, f1), textcoords="offset points", 
                   xytext=(0,-20), ha='center', fontweight='bold', fontsize=13, color=colors[1])
    
    # Add improvement statistics
    acc_improvement = ((89.70 - 87.21) / 87.21) * 100
    f1_improvement = ((89.53 - 87.22) / 87.22) * 100
    
    # Add text box with statistics
    stats_text = f"""Performance Improvements (10→20 Epochs):
    
Accuracy: +{acc_improvement:.2f}% ({87.21:.2f}% → {89.70:.2f}%)
F1-Score: +{f1_improvement:.2f}% ({87.22:.2f}% → {89.53:.2f}%)

Training Time: 1.1h → 2.2h (+100%)
Model Size: 38.9MB (38.9M parameters)
GPU Memory: 5.1GB"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    # Add performance benchmark lines
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, linewidth=2, label='90% Benchmark')
    ax.axhline(y=85, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='85% Baseline')
    
    # Add trend arrows
    ax.annotate('', xy=(20, 89.7), xytext=(15, 88.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(17.5, 89.0, 'Improvement\nTrend', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='green',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('hybrid_v2_detailed_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating Hybrid V2 Accuracy vs F1-Score Curves...")
    create_hybrid_v2_curves()
    print("Charts generated successfully!")
    print("Files created:")
    print("  - hybrid_v2_accuracy_f1_curves.png")
    print("  - hybrid_v2_detailed_curves.png")
