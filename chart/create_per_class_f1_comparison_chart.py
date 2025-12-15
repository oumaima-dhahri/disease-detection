"""
Per-Class F1-Score Comparison Chart
Creates a bar chart diagram comparing F1-scores across all models
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data from per-class analysis
classes = [
    'yellow_rust', 'army_worm', 'fusarium_head_blight', 'septoria',
    'healthy', 'common_rust', 'brown_rust', 'aphid',
    'powdery_mildew_leaf', 'black_rust', 'leaf_blight', 'tan_spot'
]

convnext_f1 = [100.00, 98.82, 98.59, 96.47, 95.89, 98.11, 95.56, 91.95, 90.74, 91.30, 68.89, 63.89]
hybrid_cnn_vit_f1 = [100.00, 97.67, 97.22, 95.35, 95.89, 97.20, 94.51, 91.57, 92.73, 85.11, 72.94, 61.11]
hybrid_v2_f1 = [100.00, 96.55, 94.44, 95.24, 93.79, 92.86, 94.25, 92.86, 92.45, 84.78, 64.20, 68.29]
yolov9_effnet_f1 = [100.00, 96.55, 94.44, 96.39, 94.52, 92.31, 96.55, 89.41, 91.43, 89.11, 63.41, 62.50]

# Convert to numpy arrays
y_pos = np.arange(len(classes))
height = 0.2  # Height of bars for horizontal chart

# Create figure with horizontal bars
plt.style.use('default')
fig, ax = plt.subplots(figsize=(12, 14))
fig.patch.set_facecolor('white')

# Create horizontal bars with nude colors
bars1 = ax.barh(y_pos - 1.5*height, convnext_f1, height, label='ConvNeXt', 
                color='#D4C5B9', alpha=0.9, edgecolor='white', linewidth=1.5)
bars2 = ax.barh(y_pos - 0.5*height, hybrid_cnn_vit_f1, height, label='Hybrid CNN-ViT', 
                color='#D4A5A5', alpha=0.9, edgecolor='white', linewidth=1.5)
bars3 = ax.barh(y_pos + 0.5*height, hybrid_v2_f1, height, label='Hybrid V2', 
                color='#C97D60', alpha=0.9, edgecolor='white', linewidth=1.5)
bars4 = ax.barh(y_pos + 1.5*height, yolov9_effnet_f1, height, label='YOLOv9+EfficientNet', 
                color='#B8A082', alpha=0.9, edgecolor='white', linewidth=1.5)

# Customize axes with cleaner style
ax.set_yticks(y_pos)
ax.set_yticklabels(classes, fontsize=11)
ax.set_xlabel('F1-Score (%)', fontsize=16, fontweight='bold', color='#333', labelpad=15)
ax.set_ylabel('Disease Classes', fontsize=16, fontweight='bold', color='#333', labelpad=15)
ax.set_title('Per-Class F1-Score Comparison Across Models', 
             fontsize=18, fontweight='bold', pad=25, color='#2c3e50')
ax.set_xlim(50, 105)
ax.legend(loc='lower right', fontsize=12, framealpha=0.95, shadow=True, 
          fancybox=True, frameon=True, edgecolor='#ddd')
ax.grid(True, alpha=0.2, axis='x', linestyle='-', linewidth=0.5, color='#999')
ax.set_axisbelow(True)

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

# Add vertical reference lines (subtle nude tones)
ax.axvline(x=95, color='#A68B5B', linestyle='--', alpha=0.4, linewidth=1, zorder=0)
ax.axvline(x=85, color='#C9A961', linestyle='--', alpha=0.4, linewidth=1, zorder=0)
ax.axvline(x=70, color='#B87D6B', linestyle='--', alpha=0.4, linewidth=1, zorder=0)

# Add value labels (cleaner)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        width = bar.get_width()
        if width < 95:  # Only label bars below 95%
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%', ha='left', va='center', 
                   fontsize=8, fontweight='bold', color='#555')

plt.tight_layout()

# Save in the chart directory (same directory as the script)
output_path = 'per_class_f1_comparison_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Chart saved as: {output_path}")

plt.show()

print("\n" + "="*60)
print("Summary:")
print(f"  - Total classes: {len(classes)}")
print(f"  - Classes with F1 ≥ 95% (ConvNeXt): {sum(1 for c in convnext_f1 if c >= 95)}")
print(f"  - Classes with F1 < 70% (ConvNeXt): {sum(1 for c in convnext_f1 if c < 70)}")
print("="*60)

