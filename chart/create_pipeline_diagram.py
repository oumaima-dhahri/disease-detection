#!/usr/bin/env python3
"""
Create Visual Pipeline Diagram for Wheat Disease Detection
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np

def create_visual_pipeline():
    """Create a clean visual pipeline diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'data': '#E3F2FD',      # Light blue
        'model': '#FFF3E0',     # Light orange
        'train': '#E8F5E8',     # Light green
        'eval': '#FCE4EC',      # Light pink
        'interpret': '#F3E5F5'   # Light purple
    }
    
    # Main pipeline stages
    stages = [
        {
            'name': 'DATA\nPREPROCESSING',
            'x': 1, 'y': 7, 'width': 2, 'height': 1.5,
            'color': colors['data'],
            'details': ['12 Classes', 'Split 70/15/15', 'Augment', 'Normalize']
        },
        {
            'name': 'MODEL\nARCHITECTURE',
            'x': 4, 'y': 7, 'width': 2, 'height': 1.5,
            'color': colors['model'],
            'details': ['ConvNeXt', 'SC-ConvNeXt', 'Hybrid CNN-ViT', 'Hybrid V2', 'YOLOv9+EfficientNet', 'ProtoPNet']
        },
        {
            'name': 'TRAINING',
            'x': 7, 'y': 7, 'width': 2, 'height': 1.5,
            'color': colors['train'],
            'details': ['Mixed Precision', 'Progressive', 'Early Stop', 'Checkpoint']
        },
        {
            'name': 'EVALUATION',
            'x': 10, 'y': 7, 'width': 2, 'height': 1.5,
            'color': colors['eval'],
            'details': ['Accuracy/F1', 'AUC-ROC/PR', 'Cohen\'s Kappa', 'MCC', 'Statistical Tests']
        },
        {
            'name': 'INTERPRETABILITY',
            'x': 4, 'y': 4, 'width': 2, 'height': 1.5,
            'color': colors['interpret'],
            'details': ['Grad-CAM', 'Saliency Maps', 'Prototype Analysis']
        }
    ]
    
    # Draw main stages
    for stage in stages:
        # Main box
        box = FancyBboxPatch(
            (stage['x'], stage['y']), 
            stage['width'], stage['height'],
            boxstyle="round,pad=0.1",
            facecolor=stage['color'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        
        # Stage title
        ax.text(stage['x'] + stage['width']/2, stage['y'] + stage['height'] - 0.3, 
                stage['name'], ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Stage details
        for i, detail in enumerate(stage['details'][:4]):  # Show max 4 details
            ax.text(stage['x'] + 0.1, stage['y'] + stage['height'] - 0.5 - i*0.2, 
                    f"• {detail}", ha='left', va='center', 
                    fontsize=8)
    
    # Add arrows showing flow
    arrows = [
        # Main flow
        ((3, 7.75), (4, 7.75)),      # Data to Model
        ((6, 7.75), (7, 7.75)),      # Model to Training
        ((9, 7.75), (10, 7.75)),     # Training to Evaluation
        # Training to Interpretability
        ((8, 7), (5, 5.5)),
        # Evaluation to Interpretability
        ((10, 7), (6, 5.5))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    # Add pipeline title
    ax.text(6, 9.5, 'WHEAT DISEASE DETECTION PIPELINE', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add technical specs box
    tech_box = FancyBboxPatch(
        (1, 1), 10, 1.5,
        boxstyle="round,pad=0.1",
        facecolor='#F5F5F5',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(tech_box)
    
    ax.text(6, 2.2, 'TECHNICAL INFRASTRUCTURE', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    tech_details = [
        'Hardware: NVIDIA Tesla T4 (16GB) | Intel Xeon E5-2686 v4 | 64GB RAM',
        'Software: PyTorch 1.12.1 | CUDA 11.8 | OpenCV 4.6.0 | NumPy 1.21.6',
        'Reproducibility: Fixed Seeds | CUDA Determinism | Algorithm Determinism'
    ]
    
    for i, detail in enumerate(tech_details):
        ax.text(1.1, 1.8 - i*0.2, detail, ha='left', va='center', 
                fontsize=9)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['data'], label='Data Preprocessing'),
        patches.Patch(color=colors['model'], label='Model Architecture'),
        patches.Patch(color=colors['train'], label='Training'),
        patches.Patch(color=colors['eval'], label='Evaluation'),
        patches.Patch(color=colors['interpret'], label='Interpretability')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('wheat_disease_pipeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('wheat_disease_pipeline.pdf', bbox_inches='tight')
    plt.show()
    
    print("Pipeline diagram saved as:")
    print("- wheat_disease_pipeline.png")
    print("- wheat_disease_pipeline.pdf")

if __name__ == "__main__":
    create_visual_pipeline()










