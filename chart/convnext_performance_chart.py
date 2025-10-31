import matplotlib.pyplot as plt
import numpy as np

# ConvNeXt Performance Data
epochs = [10, 20]
accuracy = [90.41, 91.47]
f1_score = [89.99, 90.85]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Create smooth curves using interpolation
epochs_smooth = np.linspace(10, 20, 100)
accuracy_smooth = np.interp(epochs_smooth, epochs, accuracy)
f1_score_smooth = np.interp(epochs_smooth, epochs, f1_score)

# Plot curves
line1 = ax.plot(epochs_smooth, accuracy_smooth, label='Accuracy (%)', 
                color='#2E86AB', linewidth=3, marker='o', markersize=8, 
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2E86AB')
line2 = ax.plot(epochs_smooth, f1_score_smooth, label='F1-Score (%)', 
                color='#A23B72', linewidth=3, marker='s', markersize=8,
                markerfacecolor='white', markeredgewidth=2, markeredgecolor='#A23B72')

# Add data points
ax.scatter(epochs, accuracy, color='#2E86AB', s=100, zorder=5, edgecolors='black', linewidth=2)
ax.scatter(epochs, f1_score, color='#A23B72', s=100, zorder=5, edgecolors='black', linewidth=2)

# Customize the chart
ax.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
ax.set_title('ConvNeXt Performance Progression: 10 vs 20 Epochs', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(9, 21)
ax.set_ylim(89, 92.5)

# Set x-axis ticks
ax.set_xticks([10, 20])
ax.set_xticklabels(['10 Epochs', '20 Epochs'])

# Add value labels on data points
for i, (epoch, acc, f1) in enumerate(zip(epochs, accuracy, f1_score)):
    ax.annotate(f'{acc:.2f}%', (epoch, acc), textcoords="offset points", 
                xytext=(0,10), ha='center', fontweight='bold', fontsize=11)
    ax.annotate(f'{f1:.2f}%', (epoch, f1), textcoords="offset points", 
                xytext=(0,-15), ha='center', fontweight='bold', fontsize=11)

# Add improvement indicators
improvement_acc = accuracy[1] - accuracy[0]
improvement_f1 = f1_score[1] - f1_score[0]

# Add improvement arrows
ax.annotate(f'Accuracy\n+{improvement_acc:.2f}%', 
            xy=(20, accuracy[1]), xytext=(18.5, accuracy[1] + 0.3),
            ha='center', va='center', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

ax.annotate(f'F1-Score\n+{improvement_f1:.2f}%', 
            xy=(20, f1_score[1]), xytext=(18.5, f1_score[1] - 0.3),
            ha='center', va='center', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

# Add legend
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Add performance improvement text box
textstr = f'Performance Improvement:\nAccuracy: +{improvement_acc:.2f}%\nF1-Score: +{improvement_f1:.2f}%\nTraining Time: +70% (1.0h → 1.7h)'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props, fontweight='bold')

# Customize spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Add subtle background color
ax.set_facecolor('#f8f9fa')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('convnext_performance_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('convnext_performance_curves.pdf', bbox_inches='tight')

# Show the plot
plt.show()

print("Curve chart saved as 'convnext_performance_curves.png' and 'convnext_performance_curves.pdf'")
