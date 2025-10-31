#!/usr/bin/env python3
"""
Comprehensive Architecture Comparison Diagram
Shows all 6 models in a single comparison view
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np

def create_comparison_diagram():
    """Create a comprehensive comparison diagram of all 6 architectures"""
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(10, 19, "Deep Learning Architecture Comparison for Wheat Disease Detection", 
            fontsize=24, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    # Define architectures with their key characteristics
    architectures = {
        "ConvNeXt": {
            "pos": (1, 15), "size": (3, 4), "color": "#FF6B6B",
            "features": ["Modern CNN", "LayerNorm", "7×7 Conv", "GELU", "28M params"]
        },
        "SC-ConvNeXt": {
            "pos": (5, 15), "size": (3, 4), "color": "#4ECDC4", 
            "features": ["ConvNeXt + CBAM", "Channel Attention", "Spatial Attention", "LeakyReLU", "32M params"]
        },
        "Hybrid CNN-ViT": {
            "pos": (9, 15), "size": (3, 4), "color": "#45B7D1",
            "features": ["CNN + Transformer", "12 Attention Heads", "768 Dim", "Attention Fusion", "46M params"]
        },
        "Hybrid V2": {
            "pos": (13, 15), "size": (3, 4), "color": "#96CEB4",
            "features": ["Enhanced Fusion", "Cross-Modal", "Adaptive Weighting", "Progressive", "39M params"]
        },
        "YOLOv9+EfficientNet": {
            "pos": (17, 15), "size": (3, 4), "color": "#FFEAA7",
            "features": ["Detection Adapted", "EfficientNet-B3", "PANet", "Grid-based", "52M params"]
        },
        "ProtoPNet": {
            "pos": (1, 10), "size": (3, 4), "color": "#DDA0DD",
            "features": ["Interpretable", "VGG-19", "120 Prototypes", "L2 Distance", "15M params"]
        }
    }
    
    # Draw each architecture
    for name, info in architectures.items():
        x, y = info["pos"]
        w, h = info["size"]
        color = info["color"]
        
        # Main box
        box = FancyBboxPatch((x, y), w, h, 
                           boxstyle="round,pad=0.2", 
                           facecolor=color, 
                           edgecolor='black', 
                           linewidth=2,
                           alpha=0.8)
        ax.add_patch(box)
        
        # Title
        ax.text(x + w/2, y + h - 0.3, name, fontsize=14, fontweight='bold', 
                ha='center', va='center', color='white')
        
        # Features
        for i, feature in enumerate(info["features"]):
            ax.text(x + 0.1, y + h - 0.7 - i*0.3, f"• {feature}", 
                   fontsize=10, ha='left', va='center', color='white')
    
    # Add performance comparison
    performance_data = {
        "ConvNeXt": {"Accuracy": "91.47%", "F1": "90.85%", "Time": "1.7h"},
        "SC-ConvNeXt": {"Accuracy": "91.47%", "F1": "91.42%", "Time": "2.9h"},
        "Hybrid CNN-ViT": {"Accuracy": "89.70%", "F1": "89.53%", "Time": "2.2h"},
        "Hybrid V2": {"Accuracy": "89.70%", "F1": "89.53%", "Time": "2.2h"},
        "YOLOv9+EfficientNet": {"Accuracy": "86.86%", "F1": "86.59%", "Time": "5.5h"},
        "ProtoPNet": {"Accuracy": "69.98%", "F1": "70.84%", "Time": "2.1h"}
    }
    
    # Performance table
    ax.text(6, 8, "Performance Comparison (20 Epochs)", fontsize=16, fontweight='bold', ha='center')
    
    y_pos = 7
    ax.text(1, y_pos, "Model", fontsize=12, fontweight='bold', ha='left')
    ax.text(4, y_pos, "Accuracy", fontsize=12, fontweight='bold', ha='center')
    ax.text(6, y_pos, "F1-Score", fontsize=12, fontweight='bold', ha='center')
    ax.text(8, y_pos, "Training Time", fontsize=12, fontweight='bold', ha='center')
    
    for name, perf in performance_data.items():
        y_pos -= 0.4
        ax.text(1, y_pos, name, fontsize=10, ha='left')
        ax.text(4, y_pos, perf["Accuracy"], fontsize=10, ha='center')
        ax.text(6, y_pos, perf["F1"], fontsize=10, ha='center')
        ax.text(8, y_pos, perf["Time"], fontsize=10, ha='center')
    
    # Add arrows showing data flow
    ax.text(10, 5, "Data Flow: Input Image → Feature Extraction → Classification → Disease Prediction", 
            fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    # Add key insights
    insights = [
        "Key Insights:",
        "• ConvNeXt & SC-ConvNeXt: Best accuracy (91.47%)",
        "• ProtoPNet: Most interpretable but lower accuracy",
        "• Hybrid models: Good balance of performance",
        "• YOLOv9: Longest training time but spatial awareness"
    ]
    
    ax.text(12, 3, "\n".join(insights), fontsize=11, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('architecture_comparison_comprehensive.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# Generate the comprehensive comparison
print("Creating comprehensive architecture comparison diagram...")
create_comparison_diagram()
print("Comprehensive comparison diagram created: architecture_comparison_comprehensive.png")

print("\nSummary of all created diagrams:")
print("1. Basic architecture diagrams (6 files)")
print("2. Detailed architecture diagrams (6 files)")  
print("3. Comprehensive comparison diagram (1 file)")
print("\nTotal: 13 architecture diagrams created!")

