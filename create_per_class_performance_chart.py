import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Data from your table
data = {
    'Class': [
        'Aphid', 'Army Worm', 'Black Rust', 'Brown Rust', 'Common Rust',
        'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 
        'Powdery Mildew Leaf', 'Spetoria', 'Tan Spot', 'Yellow Rust'
    ],
    'Precision': [0.977, 1.000, 0.932, 0.955, 0.962, 1.000, 1.000, 0.756, 
                  0.962, 0.976, 0.574, 1.000],
    'Recall': [0.955, 0.977, 0.891, 0.955, 0.962, 0.971, 0.944, 0.723, 
               0.944, 1.000, 0.750, 1.000],
    'F1-Score': [0.966, 0.988, 0.911, 0.955, 0.962, 0.986, 0.971, 0.739, 
                 0.953, 0.988, 0.651, 1.000]
}

# Create figure
fig, ax = plt.subplots(figsize=(18, 8))

# Set up bar positions
x = np.arange(len(data['Class']))
width = 0.25

# Create bars with colors matching your image EXACTLY
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

# Customize the plot to match image exactly
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

# Remove annotations to match the image exactly

plt.tight_layout()
plt.savefig('per_class_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('per_class_performance_comparison.pdf', bbox_inches='tight')
plt.show()

print("Charts saved:")
print("   - per_class_performance_comparison.png (300 DPI)")
print("   - per_class_performance_comparison.pdf (vector format)")

