import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Data from classification report (exact values from image)
data = {
    'Class': [
        'aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust',
        'fusarium_head_blight', 'healthy', 'leaf_blight', 
        'powdery_mildew_leaf', 'spetoria', 'tan_spot', 'yellow_rust'
    ],
    'Precision': [0.9333, 1.0, 0.9286, 0.9556, 0.9455, 1.0, 1.0, 0.8056, 
                  0.9412, 0.9318, 0.5577, 1.0],
    'Recall': [0.9545, 0.9767, 0.8478, 0.9773, 0.9811, 1.0, 0.9444, 0.617, 
               0.8889, 1.0, 0.8056, 1.0],
    'F1-Score': [0.9438, 0.9882, 0.8864, 0.9663, 0.963, 1.0, 0.9714, 0.6988, 
                 0.9143, 0.9647, 0.6591, 1.0]
}

# Create figure
fig, ax = plt.subplots(figsize=(18, 8))

# Set up bar positions
x = np.arange(len(data['Class']))
width = 0.25

# Create bars with colors matching image exactly
# Precision: Muted blue-grey (#5F819D), Recall: Lighter blue (#9AC0D9), F1-Score: Muted teal (#70B8B0)
bars1 = ax.bar(x - width, data['Precision'], width, 
               label='Precision', color='#5F819D', alpha=0.9, edgecolor='none')
bars2 = ax.bar(x, data['Recall'], width, 
               label='Recall', color='#9AC0D9', alpha=0.9, edgecolor='none')
bars3 = ax.bar(x + width, data['F1-Score'], width, 
               label='F1-Score', color='#70B8B0', alpha=0.9, edgecolor='none')

# Add reference lines (matching image exactly: green #66BB6A at 0.95, yellow #FFD54F at 0.85)
ax.axhline(y=0.95, color='#66BB6A', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=0.85, color='#FFD54F', linestyle='--', linewidth=2, alpha=0.7)

# Customize the plot
ax.set_xlabel('Disease Classes', fontsize=11, fontweight='bold')
ax.set_ylabel('Score', fontsize=11, fontweight='bold')
ax.set_title('Per-Class Performance Metrics Comparison', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(data['Class'], rotation=45, ha='right', fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig('classification_report_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('classification_report_chart.pdf', bbox_inches='tight')
plt.show()

print("Charts saved:")
print("   - classification_report_chart.png (300 DPI)")
print("   - classification_report_chart.pdf (vector format)")

