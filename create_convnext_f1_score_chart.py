import matplotlib.pyplot as plt
import numpy as np

# F1-Score data from ConvNext classification report (Epoch 20)
classification_data = {
    'aphid': 0.9195,
    'army_worm': 0.9882,
    'black_rust': 0.9130,
    'brown_rust': 0.9556,
    'common_rust': 0.9811,
    'fusarium_head_blight': 0.9859,
    'healthy': 0.9589,
    'leaf_blight': 0.6889,
    'powdery_mildew_leaf': 0.9074,
    'spetoria': 0.9647,
    'tan_spot': 0.6389,
    'yellow_rust': 1.0000
}

# Sort by F1-score for better visualization
sorted_data = sorted(classification_data.items(), key=lambda x: x[1], reverse=True)
classes = [item[0].replace('_', ' ').title() for item in sorted_data]
f1_scores = [item[1] for item in sorted_data]

# Color coding: Green for excellent (>0.9), Orange for good (0.7-0.9), Red for needs improvement (<0.7)
colors = []
for score in f1_scores:
    if score >= 0.9:
        colors.append('#2ecc71')  # Green
    elif score >= 0.7:
        colors.append('#f39c12')  # Orange
    else:
        colors.append('#e74c3c')  # Red

# Create the figure
fig, ax = plt.subplots(figsize=(12, 8))

# Create horizontal bar chart
bars = ax.barh(classes, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    width = bar.get_width()
    ax.text(width + 0.01, i, f'{score:.4f}', 
            va='center', fontweight='bold', fontsize=10)

# Customize the plot
ax.set_xlabel('F1-Score', fontsize=14, fontweight='bold')
ax.set_title('Per-Class F1-Score Performance - ConvNext Model', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 1.15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend for color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Excellent (≥0.9)'),
    Patch(facecolor='#f39c12', label='Good (0.7-0.9)'),
    Patch(facecolor='#e74c3c', label='Needs Improvement (<0.7)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

# Add average line
macro_avg = 0.9085
weighted_avg = 0.9132
ax.axvline(x=macro_avg, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Macro Avg: {macro_avg:.4f}')
ax.axvline(x=weighted_avg, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Weighted Avg: {weighted_avg:.4f}')
ax.legend(handles=legend_elements + [
    plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label=f'Macro Avg: {macro_avg:.4f}'),
    plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=2, label=f'Weighted Avg: {weighted_avg:.4f}')
], loc='lower right', fontsize=10)

# Add statistics text box
stats_text = f'Macro Average: {macro_avg:.4f}\nWeighted Average: {weighted_avg:.4f}\nTotal Classes: {len(classes)}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the figure
output_file = 'convnext_f1_score_bar_chart_epoch20.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] F1-Score bar chart saved as: {output_file}")

# Print summary
print("\n" + "="*60)
print("F1-Score Performance Summary:")
print("="*60)
print(f"  Best performing class: {classes[0]} ({f1_scores[0]:.4f})")
print(f"  Worst performing class: {classes[-1]} ({f1_scores[-1]:.4f})")
print(f"  Macro Average: {macro_avg:.4f}")
print(f"  Weighted Average: {weighted_avg:.4f}")
print("="*60)

plt.show()

