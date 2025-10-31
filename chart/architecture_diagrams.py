#!/usr/bin/env python3
"""
Quick Architecture Diagram Generator for 6 Models
Creates visual diagrams for ConvNeXt, SC-ConvNeXt, Hybrid CNN-ViT, Hybrid V2, YOLOv9+EfficientNet, ProtoPNet
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np

def create_architecture_diagram(model_name, components, figsize=(12, 8)):
    """Create a single architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, f"{model_name} Architecture", fontsize=16, fontweight='bold', ha='center')
    
    # Draw components
    y_pos = 8.5
    for i, component in enumerate(components):
        # Draw box
        box = FancyBboxPatch((1, y_pos-0.3), 8, 0.6, 
                           boxstyle="round,pad=0.1", 
                           facecolor='lightblue', 
                           edgecolor='navy', 
                           linewidth=2)
        ax.add_patch(box)
        
        # Add text
        ax.text(5, y_pos, component, fontsize=12, ha='center', va='center', fontweight='bold')
        
        # Add arrow
        if i < len(components) - 1:
            arrow = Arrow(5, y_pos-0.4, 0, -0.6, width=0.1, facecolor='red', edgecolor='red')
            ax.add_patch(arrow)
        
        y_pos -= 1.2
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace("+", "_").replace(" ", "_")}_architecture.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Define all architectures
architectures = {
    "ConvNeXt": [
        "Input Image (224×224×3)",
        "Stem Layer (4×4 Conv)",
        "Stage 1: ConvNeXt Blocks (96 channels)",
        "Stage 2: ConvNeXt Blocks (192 channels)", 
        "Stage 3: ConvNeXt Blocks (384 channels)",
        "Stage 4: ConvNeXt Blocks (768 channels)",
        "Global Average Pooling",
        "LayerNorm + Linear(768→12)",
        "Output: 12 Disease Classes"
    ],
    
    "SC-ConvNeXt": [
        "Input Image (224×224×3)",
        "ConvNeXt Backbone (Stages 1-4)",
        "CBAM Module (Channel + Spatial Attention)",
        "Global Average Pooling",
        "LayerNorm + Linear(384→12)",
        "Output: 12 Disease Classes"
    ],
    
    "Hybrid CNN-ViT": [
        "Input Image (224×224×3)",
        "CNN Backbone (ResNet layers 1-4)",
        "Patch Embedding",
        "Transformer Encoder (12 heads, 768 dim)",
        "Attention Fusion Module",
        "Classification Head",
        "Output: 12 Disease Classes"
    ],
    
    "Hybrid V2": [
        "Input Image (224×224×3)",
        "CNN Branch (ResNet)",
        "ViT Branch (Transformer)",
        "Cross-Modal Attention",
        "Adaptive Feature Fusion",
        "Progressive Integration",
        "Classification Head",
        "Output: 12 Disease Classes"
    ],
    
    "YOLOv9+EfficientNet": [
        "Input Image (224×224×3)",
        "Feature Extractor (64→128→256 channels)",
        "EfficientNet-B3 Backbone",
        "PANet Neck",
        "Global Average Pooling",
        "Classifier (512→256→128→12)",
        "Output: 12 Disease Classes"
    ],
    
    "ProtoPNet": [
        "Input Image (224×224×3)",
        "VGG-19 Backbone",
        "Feature Extraction",
        "Prototype Layer (120 prototypes)",
        "Distance Computation (L2)",
        "Similarity Calculation",
        "Classification Head",
        "Output: 12 Disease Classes"
    ]
}

# Generate all diagrams
print("Creating architecture diagrams...")
for model_name, components in architectures.items():
    print(f"Generating {model_name} diagram...")
    create_architecture_diagram(model_name, components)

print("All architecture diagrams created successfully!")
print("Files saved:")
for model_name in architectures.keys():
    filename = f'{model_name.lower().replace("+", "_").replace(" ", "_")}_architecture.png'
    print(f"  - {filename}")
