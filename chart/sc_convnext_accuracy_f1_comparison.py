#!/usr/bin/env python3
"""
SC-ConvNeXt Accuracy vs F1-Score Comparison Chart
=================================================
This script creates a focused comparison chart for Accuracy vs F1-Score
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_accuracy_f1_comparison():
    """Create focused Accuracy vs F1-Score comparison chart"""
    
    # Data
    configurations = ['10 Epochs', '20 Epochs']
    accuracy = [88.10, 91.47]
    f1_score = [87.50, 91.42]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('SC-ConvNeXt: Accuracy vs F1-Score Comparison', fontsize=18, fontweight='bold')
    
    # Colors
    colors = ['#2E86AB', '#A23B72']
    
    # Chart 1: Side-by-side comparison
    x = np.arange(len(configurations))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy (%)', 
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, f1_score, width, label='F1-Score (%)', 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_title('Performance Metrics Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Training Configuration', fontsize=12)
    ax1.set_ylabel('Performance (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configurations)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(85, 93)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Chart 2: Line plot showing improvement
    ax2.plot(configurations, accuracy, 'o-', color=colors[0], linewidth=3, 
             markersize=10, label='Accuracy (%)', alpha=0.8)
    ax2.plot(configurations, f1_score, 's-', color=colors[1], linewidth=3, 
             markersize=10, label='F1-Score (%)', alpha=0.8)
    
    ax2.set_title('Performance Improvement Trend', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Training Configuration', fontsize=12)
    ax2.set_ylabel('Performance (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(85, 93)
    
    # Add value labels on points
    for i, (acc, f1) in enumerate(zip(accuracy, f1_score)):
        ax2.text(i, acc + 0.3, f'{acc:.2f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=11, color=colors[0])
        ax2.text(i, f1 - 0.5, f'{f1:.2f}%', ha='center', va='top', 
                fontweight='bold', fontsize=11, color=colors[1])
    
    # Add improvement annotations
    acc_improvement = ((91.47 - 88.10) / 88.10) * 100
    f1_improvement = ((91.42 - 87.50) / 87.50) * 100
    
    ax2.annotate(f'+{acc_improvement:.1f}%', xy=(0.5, 89.5), xytext=(0.5, 90.5),
                arrowprops=dict(arrowstyle='->', color=colors[0], lw=2),
                fontsize=12, fontweight='bold', color=colors[0], ha='center')
    
    ax2.annotate(f'+{f1_improvement:.1f}%', xy=(0.5, 88.5), xytext=(0.5, 87.5),
                arrowprops=dict(arrowstyle='->', color=colors[1], lw=2),
                fontsize=12, fontweight='bold', color=colors[1], ha='center')
    
    plt.tight_layout()
    plt.savefig('sc_convnext_accuracy_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed single chart
    create_detailed_comparison()

def create_detailed_comparison():
    """Create a detailed single chart with more information"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Data
    configurations = ['10 Epochs', '20 Epochs']
    accuracy = [88.10, 91.47]
    f1_score = [87.50, 91.42]
    
    # Colors
    colors = ['#2E86AB', '#A23B72']
    
    # Create grouped bar chart
    x = np.arange(len(configurations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', 
                   color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, f1_score, width, label='F1-Score (%)', 
                   color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Customize chart
    ax.set_title('SC-ConvNeXt: Accuracy vs F1-Score Performance Comparison', 
                fontweight='bold', fontsize=16, pad=20)
    ax.set_xlabel('Training Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configurations, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(85, 93)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    # Add improvement statistics
    acc_improvement = ((91.47 - 88.10) / 88.10) * 100
    f1_improvement = ((91.42 - 87.50) / 87.50) * 100
    
    # Add text box with statistics
    stats_text = f"""Performance Improvements (10→20 Epochs):
    
Accuracy: +{acc_improvement:.2f}% ({88.10:.2f}% → {91.47:.2f}%)
F1-Score: +{f1_improvement:.2f}% ({87.50:.2f}% → {91.42:.2f}%)

Training Time: 0.9h → 2.9h (+222%)
Model Size: 32.1MB (32.1M parameters)
GPU Memory: 4.8GB"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Add performance lines
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% Benchmark')
    ax.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='85% Baseline')
    
    plt.tight_layout()
    plt.savefig('sc_convnext_accuracy_f1_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating SC-ConvNeXt Accuracy vs F1-Score Comparison Charts...")
    create_accuracy_f1_comparison()
    print("Charts generated successfully!")
    print("Files created:")
    print("  - sc_convnext_accuracy_f1_comparison.png")
    print("  - sc_convnext_accuracy_f1_detailed.png")

