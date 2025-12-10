import matplotlib.pyplot as plt
import numpy as np

# SC-ConvNext classification report data (Epoch 20)
classification_data = {
    'aphid': {'precision': 1.0000, 'recall': 0.9318, 'f1': 0.9647, 'support': 44},
    'army_worm': {'precision': 1.0000, 'recall': 1.0000, 'f1': 1.0000, 'support': 43},
    'black_rust': {'precision': 0.9091, 'recall': 0.8696, 'f1': 0.8889, 'support': 46},
    'brown_rust': {'precision': 0.9545, 'recall': 0.9545, 'f1': 0.9545, 'support': 44},
    'common_rust': {'precision': 0.9636, 'recall': 1.0000, 'f1': 0.9815, 'support': 53},
    'fusarium_head_blight': {'precision': 0.9459, 'recall': 1.0000, 'f1': 0.9722, 'support': 35},
    'healthy': {'precision': 0.9861, 'recall': 0.9861, 'f1': 0.9861, 'support': 72},
    'leaf_blight': {'precision': 0.7500, 'recall': 0.5745, 'f1': 0.6506, 'support': 47},
    'powdery_mildew_leaf': {'precision': 0.9074, 'recall': 0.9074, 'f1': 0.9074, 'support': 54},
    'spetoria': {'precision': 0.9318, 'recall': 1.0000, 'f1': 0.9647, 'support': 41},
    'tan_spot': {'precision': 0.5652, 'recall': 0.7027, 'f1': 0.6265, 'support': 37},
    'yellow_rust': {'precision': 1.0000, 'recall': 1.0000, 'f1': 1.0000, 'support': 47}
}

# Extract data
classes = list(classification_data.keys())
precisions = [classification_data[cls]['precision'] for cls in classes]
recalls = [classification_data[cls]['recall'] for cls in classes]
f1_scores = [classification_data[cls]['f1'] for cls in classes]
supports = [classification_data[cls]['support'] for cls in classes]

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Scale bubble size by support (multiply by factor for visibility)
bubble_sizes = [s * 3 for s in supports]

# Create scatter plot with color coding by F1-score
scatter = ax.scatter(recalls, precisions, s=bubble_sizes, c=f1_scores, 
                     cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=2, 
                     vmin=0.5, vmax=1.0)

# Add class labels
for i, cls in enumerate(classes):
    # Format class name for better display
    display_name = cls.replace('_', '\n')
    ax.annotate(display_name, 
                (recalls[i], precisions[i]), 
                fontsize=9, ha='center', va='center', 
                fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

# Customize the plot
ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
ax.set_title('Precision vs Recall - SC-ConvNext Model', fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0.5, 1.05)
ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3, linestyle='--')

# Add diagonal line for reference (precision = recall)
ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.3, linewidth=1, label='Precision = Recall')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, label='F1-Score', shrink=0.8)
cbar.set_label('F1-Score', fontsize=12, fontweight='bold')

# Add legend for bubble size
from matplotlib.patches import Patch
legend_elements = [
    plt.scatter([], [], s=35*3, c='gray', alpha=0.7, edgecolors='black', linewidth=2, label='Small (35-40 samples)'),
    plt.scatter([], [], s=47*3, c='gray', alpha=0.7, edgecolors='black', linewidth=2, label='Medium (44-47 samples)'),
    plt.scatter([], [], s=72*3, c='gray', alpha=0.7, edgecolors='black', linewidth=2, label='Large (53-72 samples)')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10, title='Bubble Size = Support', 
          title_fontsize=10, framealpha=0.9)

# Add text annotation for macro and weighted averages
macro_prec = 0.9095
macro_rec = 0.9106
weighted_prec = 0.9172
weighted_rec = 0.9147

ax.scatter(macro_rec, macro_prec, s=200, marker='*', c='blue', 
           edgecolors='black', linewidth=2, label='Macro Avg', zorder=5)
ax.scatter(weighted_rec, weighted_prec, s=200, marker='*', c='purple', 
           edgecolors='black', linewidth=2, label='Weighted Avg', zorder=5)

# Update legend to include averages
legend_elements.append(plt.scatter([], [], s=200, marker='*', c='blue', 
                                    edgecolors='black', linewidth=2, label='Macro Average'))
legend_elements.append(plt.scatter([], [], s=200, marker='*', c='purple', 
                                    edgecolors='black', linewidth=2, label='Weighted Average'))
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, 
          title='Legend', title_fontsize=10, framealpha=0.9)

# Add quadrant labels
ax.text(0.75, 0.95, 'High Precision\nHigh Recall', fontsize=11, 
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax.text(0.75, 0.6, 'High Precision\nLow Recall', fontsize=11, 
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.6, 0.95, 'Low Precision\nHigh Recall', fontsize=11, 
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax.text(0.6, 0.6, 'Low Precision\nLow Recall', fontsize=11, 
        ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

plt.tight_layout()

# Save the figure
output_file = 'sc_convnext_precision_recall_scatter.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Precision vs Recall scatter plot saved as: {output_file}")

# Print summary
print("\n" + "="*60)
print("Precision vs Recall Analysis:")
print("="*60)
print(f"  Best performing (top-right): {classes[np.argmax([p + r for p, r in zip(precisions, recalls)])]}")
print(f"  Worst performing (bottom-left): {classes[np.argmin([p + r for p, r in zip(precisions, recalls)])]}")
print(f"  Macro Average: Precision={macro_prec:.4f}, Recall={macro_rec:.4f}")
print(f"  Weighted Average: Precision={weighted_prec:.4f}, Recall={weighted_rec:.4f}")
print("="*60)

plt.show()

