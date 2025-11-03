#!/usr/bin/env python3
"""
Hybrid CNN-ViT Performance Metrics Comparison
==============================================
Creates a bar chart showing Accuracy and F1-Score comparison
between 10 Epochs and 20 Epochs training configurations
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8')

def create_performance_chart():
    """Create Performance Metrics Comparison bar chart"""
    
    # Correct values based on actual training outputs
    configurations = ['10 Epochs', '20 Epochs']
    accuracy = [88.45, 90.94]  # Corrected Epoch 20 value
    f1_score = [88.35, 90.70]  # Corrected Epoch 20 value
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Set up bar chart
    x = np.arange(len(configurations))
    width = 0.35
    
    # Colors matching the style from the image
    accuracy_color = '#4A90E2'  # Teal/Blue
    f1_color = '#C085C5'        # Muted Pink/Purple
    
    # Create bars
    bars1 = ax.bar(x - width/2, accuracy, width, 
                   label='Accuracy (%)', 
                   color=accuracy_color, 
                   alpha=0.8, 
                   edgecolor='black', 
                   linewidth=1.5)
    
    bars2 = ax.bar(x + width/2, f1_score, width, 
                   label='F1-Score (%)', 
                   color=f1_color, 
                   alpha=0.8, 
                   edgecolor='black', 
                   linewidth=1.5)
    
    # Customize chart
    ax.set_title('Performance Metrics Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Training Configuration', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (%)', 
                 fontsize=14, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(configurations, fontsize=12)
    
    # Set y-axis range to match the image style
    ax.set_ylim(85, 92)
    ax.set_yticks(np.arange(85, 93, 1))
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}%',
                   ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
    
    # Add improvement annotations
    acc_improvement = accuracy[1] - accuracy[0]
    f1_improvement = f1_score[1] - f1_score[0]
    
    # Add text annotation for improvements
    improvement_text = f'Improvements:\nAccuracy: +{acc_improvement:.2f}%\nF1-Score: +{f1_improvement:.2f}%'
    ax.text(0.98, 0.15, improvement_text,
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('hybrid_cnn_vit_performance_comparison.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("Chart saved as: hybrid_cnn_vit_performance_comparison.png")
    
    # Show plot
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("Hybrid CNN-ViT Performance Summary")
    print("="*50)
    print(f"10 Epochs:  Accuracy = {accuracy[0]:.2f}%, F1-Score = {f1_score[0]:.2f}%")
    print(f"20 Epochs:  Accuracy = {accuracy[1]:.2f}%, F1-Score = {f1_score[1]:.2f}%")
    print(f"\nImprovements:")
    print(f"  Accuracy: +{acc_improvement:.2f}% ({((acc_improvement/accuracy[0])*100):.2f}% relative)")
    print(f"  F1-Score: +{f1_improvement:.2f}% ({((f1_improvement/f1_score[0])*100):.2f}% relative)")
    print("="*50)

if __name__ == "__main__":
    print("Creating Hybrid CNN-ViT Performance Metrics Comparison Chart...")
    create_performance_chart()
    print("\nChart generation complete!")

