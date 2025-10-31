#!/usr/bin/env python3
"""
Detailed Architecture Diagram Generator
Creates professional-looking architecture diagrams with detailed components
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow, Polygon
import numpy as np

def create_detailed_diagram(model_name, stages, figsize=(14, 10)):
    """Create detailed architecture diagram with better visuals"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title with styling
    ax.text(6, 11.5, f"{model_name} Architecture", 
            fontsize=20, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
    
    # Draw stages
    y_pos = 10.5
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    for i, stage in enumerate(stages):
        color = colors[i % len(colors)]
        
        # Main box
        box = FancyBboxPatch((1, y_pos-0.4), 10, 0.8, 
                           boxstyle="round,pad=0.15", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=2,
                           alpha=0.8)
        ax.add_patch(box)
        
        # Add text
        ax.text(6, y_pos, stage, fontsize=11, ha='center', va='center', 
                fontweight='bold', color='white')
        
        # Add arrow
        if i < len(stages) - 1:
            arrow = Arrow(6, y_pos-0.5, 0, -0.8, width=0.2, 
                         facecolor='darkred', edgecolor='darkred')
            ax.add_patch(arrow)
        
        y_pos -= 1.3
    
    # Add legend/notes
    ax.text(0.5, 1, f"Model: {model_name}\nParameters: See implementation details\nInput: 224×224×3 RGB images", 
            fontsize=9, ha='left', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    filename = f'{model_name.lower().replace("+", "_").replace(" ", "_")}_detailed_architecture.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return filename

# Detailed architecture definitions
detailed_architectures = {
    "ConvNeXt": [
        "Input: RGB Image (224×224×3)",
        "Stem: 4×4 Conv + LayerNorm",
        "Stage 1: ConvNeXt Blocks (96 channels, 3 blocks)",
        "Stage 2: ConvNeXt Blocks (192 channels, 3 blocks)",
        "Stage 3: ConvNeXt Blocks (384 channels, 9 blocks)",
        "Stage 4: ConvNeXt Blocks (768 channels, 3 blocks)",
        "Global Average Pooling",
        "LayerNorm + Linear Classifier (768→12)",
        "Output: Disease Classification (12 classes)"
    ],
    
    "SC-ConvNeXt": [
        "Input: RGB Image (224×224×3)",
        "ConvNeXt Backbone (Stages 1-4)",
        "CBAM Module: Channel Attention (LeakyReLU)",
        "CBAM Module: Spatial Attention (7×7 Conv)",
        "Global Average Pooling",
        "LayerNorm + Linear Classifier (384→12)",
        "Output: Disease Classification (12 classes)"
    ],
    
    "Hybrid CNN-ViT": [
        "Input: RGB Image (224×224×3)",
        "CNN Backbone: ResNet (layers 1-4)",
        "Patch Embedding: Convert to tokens",
        "Transformer Encoder: 12 heads, 768 dim",
        "Attention Fusion: Tanh activation",
        "Classification Head: Linear projection",
        "Output: Disease Classification (12 classes)"
    ],
    
    "Hybrid V2": [
        "Input: RGB Image (224×224×3)",
        "CNN Branch: ResNet feature extraction",
        "ViT Branch: Transformer processing",
        "Cross-Modal Attention: Bidirectional",
        "Adaptive Fusion: Sigmoid weighting",
        "Progressive Integration: Multi-stage",
        "Classification Head: Final projection",
        "Output: Disease Classification (12 classes)"
    ],
    
    "YOLOv9+EfficientNet": [
        "Input: RGB Image (224×224×3)",
        "Feature Extractor: Conv layers (64→128→256)",
        "EfficientNet-B3: Compound scaling backbone",
        "PANet: Path aggregation network",
        "Global Average Pooling",
        "Classifier: MLP (512→256→128→12)",
        "Output: Disease Classification (12 classes)"
    ],
    
    "ProtoPNet": [
        "Input: RGB Image (224×224×3)",
        "VGG-19 Backbone: Feature extraction",
        "Feature Maps: High-level representations",
        "Prototype Layer: 120 prototypes (10×12 classes)",
        "Distance Computation: L2 Euclidean",
        "Similarity Calculation: Gaussian kernel",
        "Classification: Prototype-based voting",
        "Output: Disease Classification (12 classes)"
    ]
}

# Generate detailed diagrams
print("Creating detailed architecture diagrams...")
created_files = []

for model_name, stages in detailed_architectures.items():
    print(f"Generating detailed {model_name} diagram...")
    filename = create_detailed_diagram(model_name, stages)
    created_files.append(filename)

print("\nAll detailed architecture diagrams created successfully!")
print("Files created:")
for filename in created_files:
    print(f"  - {filename}")

print("\nThese diagrams show:")
print("- Detailed component breakdown")
print("- Data flow through each stage")
print("- Key architectural features")
print("- Professional visual styling")

