#!/usr/bin/env python3
"""
Hybrid CNN-ViT Performance Improvement Trend Chart
==================================================
Creates a line chart showing performance improvement from 10 to 20 epochs
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')

def create_improvement_trend_chart():
    """Create Performance Improvement Trend line chart"""
    
    # Correct values based on actual training outputs
    configurations = ['10 Epochs', '20 Epochs']
    accuracy = [88.45, 90.94]  # Corrected Epoch 20 value
    f1_score = [88.35, 90.70]  # Corrected Epoch 20 value
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colors matching the style
    accuracy_color = '#2E86AB'  # Blue
    f1_color = '#A23B72'        # Pink/Purple
    
    # Create line plots with markers
    ax.plot(configurations, accuracy, 'o-', 
            color=accuracy_color, 
            linewidth=3, 
            markersize=12, 
            label='Accuracy (%)', 
            alpha=0.8,
            markerfacecolor=accuracy_color,
            markeredgecolor='black',
            markeredgewidth=1.5)
    
    ax.plot(configurations, f1_score, 's-', 
            color=f1_color, 
            linewidth=3, 
            markersize=12, 
            label='F1-Score (%)', 
            alpha=0.8,
            markerfacecolor=f1_color,
            markeredgecolor='black',
            markeredgewidth=1.5)
    
    # Customize chart
    ax.set_title('Performance Improvement Trend', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Training Configuration', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (%)', 
                 fontsize=14, fontweight='bold')
    
    # Set y-axis range
    ax.set_ylim(87.0, 91.5)
    ax.set_yticks(np.arange(87.0, 91.5, 0.5))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Calculate improvements
    acc_improvement = accuracy[1] - accuracy[0]
    f1_improvement = f1_score[1] - f1_score[0]
    
    # Add value labels on points
    for i, (acc, f1) in enumerate(zip(accuracy, f1_score)):
        ax.text(i, acc + 0.15, f'{acc:.2f}%', 
               ha='center', va='bottom',
               fontweight='bold', fontsize=11, 
               color=accuracy_color)
        ax.text(i, f1 - 0.25, f'{f1:.2f}%', 
               ha='center', va='top',
               fontweight='bold', fontsize=11, 
               color=f1_color)
    
    # Add improvement annotations with arrows
    # Accuracy improvement arrow
    mid_x = 0.5  # Midpoint between the two configurations
    mid_y_acc = (accuracy[0] + accuracy[1]) / 2
    
    ax.annotate(f'+{acc_improvement:.2f}%', 
               xy=(mid_x, mid_y_acc + 0.3), 
               xytext=(mid_x, mid_y_acc + 0.6),
               arrowprops=dict(arrowstyle='->', 
                             color=accuracy_color, 
                             lw=2.5,
                             connectionstyle='arc3,rad=0'),
               fontsize=13, 
               fontweight='bold', 
               color=accuracy_color, 
               ha='center',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor=accuracy_color,
                        linewidth=1.5))
    
    # F1-Score improvement arrow
    mid_y_f1 = (f1_score[0] + f1_score[1]) / 2
    
    ax.annotate(f'+{f1_improvement:.2f}%', 
               xy=(mid_x, mid_y_f1 - 0.3), 
               xytext=(mid_x, mid_y_f1 - 0.6),
               arrowprops=dict(arrowstyle='->', 
                             color=f1_color, 
                             lw=2.5,
                             connectionstyle='arc3,rad=0'),
               fontsize=13, 
               fontweight='bold', 
               color=f1_color, 
               ha='center',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor=f1_color,
                        linewidth=1.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('hybrid_cnn_vit_improvement_trend.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("Chart saved as: hybrid_cnn_vit_improvement_trend.png")
    
    # Show plot
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("Hybrid CNN-ViT Performance Improvement Summary")
    print("="*60)
    print(f"10 Epochs:")
    print(f"  Accuracy:  {accuracy[0]:.2f}%")
    print(f"  F1-Score:  {f1_score[0]:.2f}%")
    print(f"\n20 Epochs:")
    print(f"  Accuracy:  {accuracy[1]:.2f}%")
    print(f"  F1-Score:  {f1_score[1]:.2f}%")
    print(f"\nImprovements:")
    print(f"  Accuracy:  +{acc_improvement:.2f}% ({((acc_improvement/accuracy[0])*100):.2f}% relative)")
    print(f"  F1-Score:  +{f1_improvement:.2f}% ({((f1_improvement/f1_score[0])*100):.2f}% relative)")
    print("="*60)

if __name__ == "__main__":
    print("Creating Hybrid CNN-ViT Performance Improvement Trend Chart...")
    create_improvement_trend_chart()
    print("\nChart generation complete!")

