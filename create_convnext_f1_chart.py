import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for professional look
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'

# Classification report data from ConvNext Epoch 20
classification_data = {
    'aphid': {'f1': 0.9195, 'precision': 0.9302, 'recall': 0.9091, 'support': 44},
    'army_worm': {'f1': 0.9882, 'precision': 1.0000, 'recall': 0.9767, 'support': 43},
    'black_rust': {'f1': 0.9130, 'precision': 0.9130, 'recall': 0.9130, 'support': 46},
    'brown_rust': {'f1': 0.9556, 'precision': 0.9348, 'recall': 0.9773, 'support': 44},
    'common_rust': {'f1': 0.9811, 'precision': 0.9811, 'recall': 0.9811, 'support': 53},
    'fusarium_head_blight': {'f1': 0.9859, 'precision': 0.9722, 'recall': 1.0000, 'support': 35},
    'healthy': {'f1': 0.9589, 'precision': 0.9459, 'recall': 0.9722, 'support': 72},
    'leaf_blight': {'f1': 0.6889, 'precision': 0.7209, 'recall': 0.6596, 'support': 47},
    'powdery_mildew_leaf': {'f1': 0.9074, 'precision': 0.9074, 'recall': 0.9074, 'support': 54},
    'spetoria': {'f1': 0.9647, 'precision': 0.9318, 'recall': 1.0000, 'support': 41},
    'tan_spot': {'f1': 0.6389, 'precision': 0.6571, 'recall': 0.6216, 'support': 37},
    'yellow_rust': {'f1': 1.0000, 'precision': 1.0000, 'recall': 1.0000, 'support': 47}
}

# Sort classes by F1-score (descending)
sorted_classes = sorted(classification_data.items(), key=lambda x: x[1]['f1'], reverse=True)
classes = [cls[0].replace('_', ' ').title() for cls in sorted_classes]
f1_scores = [cls[1]['f1'] for cls in sorted_classes]

# Color coding: Green (excellent >= 0.95), Orange (good >= 0.85), Red (needs improvement < 0.85)
colors = []
for score in f1_scores:
    if score >= 0.95:
        colors.append('#2ecc71')  # Green - Excellent
    elif score >= 0.85:
        colors.append('#f39c12')  # Orange - Good
    else:
        colors.append('#e74c3c')  # Red - Needs improvement

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Create horizontal bar chart
bars = ax.barh(classes, f1_scores, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    width = bar.get_width()
    ax.text(width + 0.01, i, f'{score:.4f}', 
            va='center', fontweight='bold', fontsize=10)

# Customize the chart
ax.set_xlabel('F1-Score', fontsize=13, fontweight='bold')
ax.set_title('ConvNext Model - Per-Class F1-Score Performance (Epoch 20)', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(0, 1.15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add vertical line at 0.9 (good performance threshold)
ax.axvline(x=0.9, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Good Performance (0.9)')
ax.axvline(x=0.95, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Excellent Performance (0.95)')

# Add legend for color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', edgecolor='black', label='Excellent (≥0.95)'),
    Patch(facecolor='#f39c12', edgecolor='black', label='Good (≥0.85)'),
    Patch(facecolor='#e74c3c', edgecolor='black', label='Needs Improvement (<0.85)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

# Add summary statistics text box
macro_avg = 0.9085
weighted_avg = 0.9132
accuracy = 0.9147

textstr = f'Overall Performance:\nAccuracy: {accuracy:.4f}\nMacro Avg F1: {macro_avg:.4f}\nWeighted Avg F1: {weighted_avg:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontweight='bold')

# Improve layout
plt.tight_layout()

# Save the figure
output_file = 'convnext_f1_score_chart.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"F1-Score bar chart saved as: {output_file}")

# Also save as PDF for high-quality printing
output_file_pdf = 'convnext_f1_score_chart.pdf'
plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
print(f"F1-Score bar chart saved as: {output_file_pdf}")

plt.show()

