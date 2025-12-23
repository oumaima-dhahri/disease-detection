import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['figure.dpi'] = 300

# Data from classification report (sorted by F1-Score descending)
data = {
    'Class': [
        'yellow_rust', 'fusarium_head_blight', 'army_worm', 'healthy', 
        'common_rust', 'spetoria', 'aphid', 'brown_rust',
        'black_rust', 'powdery_mildew_leaf', 'leaf_blight', 'tan_spot'
    ],
    'F1-Score': [1.0, 1.0, 0.9882, 0.9714, 0.963, 0.9647, 0.9438, 0.9663,
                 0.8864, 0.9143, 0.6988, 0.6591]
}

# Sort by F1-Score descending
sorted_data = sorted(zip(data['Class'], data['F1-Score']), key=lambda x: x[1], reverse=True)
classes_sorted = [x[0] for x in sorted_data]
f1_scores_sorted = [x[1] for x in sorted_data]

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Create horizontal bar chart
y_pos = np.arange(len(classes_sorted))

# Color bars based on performance thresholds
colors = []
for score in f1_scores_sorted:
    if score >= 0.95:
        colors.append('#4CAF50')  # Green for excellent
    elif score >= 0.85:
        colors.append('#2196F3')  # Blue for very good
    else:
        colors.append('#F44336')  # Red for below threshold

bars = ax.barh(y_pos, f1_scores_sorted, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

# Add reference lines (no labels, minimal)
ax.axvline(x=0.95, color='green', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvline(x=0.85, color='blue', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvline(x=0.75, color='#FFA500', linestyle='--', linewidth=1.5, alpha=0.6)

# Add value labels on bars (smaller font)
for i, (bar, score) in enumerate(zip(bars, f1_scores_sorted)):
    ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
           f'{score:.4f}', 
           va='center', fontsize=8, fontweight='bold')

# Customize the plot (smaller fonts)
ax.set_xlabel('F1-Score', fontsize=10, fontweight='bold')
ax.set_ylabel('Disease Classes', fontsize=10, fontweight='bold')
ax.set_title('F1-Score Ranking by Class', fontsize=12, fontweight='bold', pad=15)
ax.set_yticks(y_pos)
ax.set_yticklabels(classes_sorted, fontsize=9)
ax.set_xlim(0, 1.1)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.95, 1.0])
ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.75', '0.85', '0.95', '1.0'], fontsize=9)
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

# Add minimal legend outside the plot area (small font)
legend_elements = [
    plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label='Excellent (0.95)'),
    plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5, label='Very Good (0.85)'),
    plt.Line2D([0], [0], color='#FFA500', linestyle='--', linewidth=1.5, label='Good (0.75)')
]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
          fontsize=8, framealpha=0.9, title='Thresholds', title_fontsize=9)

# Invert y-axis to show highest at top
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('f1_score_ranking_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('f1_score_ranking_chart.pdf', bbox_inches='tight')
plt.show()

print("Charts saved:")
print("   - f1_score_ranking_chart.png (300 DPI)")
print("   - f1_score_ranking_chart.pdf (vector format)")

