#!/usr/bin/env python3
"""
Global Pipeline Diagram for Wheat Disease Detection Project
Creates a comprehensive visualization of the complete workflow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_global_pipeline_diagram():
    """Create a comprehensive pipeline diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'preprocessing': '#E8F4FD',
        'model': '#FFF2CC',
        'training': '#D5E8D4',
        'evaluation': '#F8CECC',
        'interpretability': '#E1D5E7'
    }
    
    # Define pipeline stages
    stages = [
        {
            'name': 'Data Preprocessing',
            'x': 1, 'y': 10, 'width': 1.5, 'height': 1.5,
            'color': colors['preprocessing'],
            'details': [
                'Raw Images (12 classes)',
                'Quality Validation',
                'Stratified Split',
                'Data Augmentation',
                'Normalization',
                'DataLoader Setup'
            ]
        },
        {
            'name': 'Model Architecture',
            'x': 4, 'y': 10, 'width': 1.5, 'height': 1.5,
            'color': colors['model'],
            'details': [
                'ConvNeXt',
                'SC-ConvNeXt',
                'Hybrid CNN-ViT',
                'Hybrid V2',
                'YOLOv9+EfficientNet',
                'ProtoPNet'
            ]
        },
        {
            'name': 'Training',
            'x': 7, 'y': 10, 'width': 1.5, 'height': 1.5,
            'color': colors['training'],
            'details': [
                'Mixed Precision',
                'Progressive Training',
                'Early Stopping',
                'Checkpointing',
                'Loss Optimization',
                'Validation Monitoring'
            ]
        },
        {
            'name': 'Evaluation',
            'x': 1, 'y': 7, 'width': 1.5, 'height': 1.5,
            'color': colors['evaluation'],
            'details': [
                'Basic Metrics',
                'Advanced Metrics',
                'Statistical Testing',
                'Robustness Analysis',
                'Cross-Validation',
                'Performance Comparison'
            ]
        },
        {
            'name': 'Interpretability',
            'x': 4, 'y': 7, 'width': 1.5, 'height': 1.5,
            'color': colors['interpretability'],
            'details': [
                'Grad-CAM',
                'Integrated Gradients',
                'Saliency Maps',
                'Prototype Analysis',
                'Feature Visualization',
                'Decision Explanations'
            ]
        }
    ]
    
    # Draw stages
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
        ax.text(stage['x'] + stage['width']/2, stage['y'] + stage['height'] - 0.2, 
                stage['name'], ha='center', va='center', 
                fontsize=12, fontweight='bold')
        
        # Stage details
        for i, detail in enumerate(stage['details']):
            ax.text(stage['x'] + 0.1, stage['y'] + stage['height'] - 0.4 - i*0.15, 
                    f"• {detail}", ha='left', va='center', 
                    fontsize=9)
    
    # Add arrows showing flow
    arrows = [
        # Preprocessing to Model
        ((2.5, 10.75), (4, 10.75)),
        # Model to Training
        ((5.5, 10.75), (7, 10.75)),
        # Training to Evaluation
        ((7.75, 10), (2.5, 8.5)),
        # Training to Interpretability
        ((7.75, 9.5), (5.5, 8.5)),
        # Evaluation to Interpretability
        ((2.5, 7), (4, 7))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    # Add pipeline title
    ax.text(5, 11.5, 'Wheat Disease Detection: Global Pipeline', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add detailed metrics section
    metrics_box = FancyBboxPatch(
        (7, 7), 2.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor='#F0F0F0',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(metrics_box)
    
    ax.text(8.25, 8.2, 'Advanced Metrics', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    metrics_details = [
        'AUC-ROC, AUC-PR',
        'Cohen\'s Kappa, MCC',
        'Confidence Intervals',
        'Statistical Significance'
    ]
    
    for i, metric in enumerate(metrics_details):
        ax.text(7.1, 7.9 - i*0.15, f"• {metric}", ha='left', va='center', 
                fontsize=9)
    
    # Add hardware specs
    hardware_box = FancyBboxPatch(
        (1, 4), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor='#E6F3FF',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(hardware_box)
    
    ax.text(2.5, 5.2, 'Hardware Configuration', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    hardware_details = [
        'GPU: NVIDIA Tesla T4 (16GB)',
        'CPU: Intel Xeon E5-2686 v4',
        'RAM: 64GB DDR4 ECC',
        'Platform: Kaggle Notebooks'
    ]
    
    for i, detail in enumerate(hardware_details):
        ax.text(1.1, 4.9 - i*0.15, f"• {detail}", ha='left', va='center', 
                fontsize=9)
    
    # Add software stack
    software_box = FancyBboxPatch(
        (5, 4), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor='#FFF0E6',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(software_box)
    
    ax.text(6.5, 5.2, 'Software Stack', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    
    software_details = [
        'PyTorch 1.12.1 + CUDA 11.8',
        'OpenCV 4.6.0, PIL 9.2.0',
        'NumPy 1.21.6, SciPy 1.9.1',
        'Pandas 1.4.4, Scikit-learn 1.1.1'
    ]
    
    for i, detail in enumerate(software_details):
        ax.text(5.1, 4.9 - i*0.15, f"• {detail}", ha='left', va='center', 
                fontsize=9)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['preprocessing'], label='Data Preprocessing'),
        patches.Patch(color=colors['model'], label='Model Architecture'),
        patches.Patch(color=colors['training'], label='Training'),
        patches.Patch(color=colors['evaluation'], label='Evaluation'),
        patches.Patch(color=colors['interpretability'], label='Interpretability')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('global_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('global_pipeline_diagram.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_global_pipeline_diagram()

