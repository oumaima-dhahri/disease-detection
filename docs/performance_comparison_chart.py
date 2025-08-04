import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Model performance data
models = ['ConvNeXt', 'SC-ConvNeXt', 'ProtoPNet']
accuracies = [90.93, 88.89, 70.07]
macro_f1 = [90.08, 88.69, 68.27]
weighted_f1 = [90.34, 88.85, 69.72]

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Wheat Disease Detection: Model Performance Comparison', fontsize=16, fontweight='bold')

# 1. Overall Accuracy Comparison
bars1 = ax1.bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'])
ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# 2. F1-Score Comparison
x = np.arange(len(models))
width = 0.35

bars2 = ax2.bar(x - width/2, macro_f1, width, label='Macro F1', color='#2E86AB')
bars3 = ax2.bar(x + width/2, weighted_f1, width, label='Weighted F1', color='#A23B72')

ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('F1-Score (%)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=9)

# 3. Per-Class Performance (ConvNeXt)
classes = ['aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust', 
           'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf', 
           'spetoria', 'tan_spot', 'yellow_rust']

convnext_f1 = [94.55, 100.0, 90.38, 97.30, 85.33, 96.70, 96.91, 71.91, 94.00, 95.89, 57.97, 100.0]
sc_convnext_f1 = [83.50, 95.74, 90.00, 96.00, 86.84, 95.35, 94.95, 71.43, 88.89, 91.89, 70.59, 99.05]

x_pos = np.arange(len(classes))
width = 0.35

bars4 = ax3.bar(x_pos - width/2, convnext_f1, width, label='ConvNeXt', color='#2E86AB')
bars5 = ax3.bar(x_pos + width/2, sc_convnext_f1, width, label='SC-ConvNeXt', color='#A23B72')

ax3.set_title('Per-Class F1-Score Comparison', fontsize=14, fontweight='bold')
ax3.set_ylabel('F1-Score (%)', fontsize=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(classes, rotation=45, ha='right')
ax3.legend()
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3)

# 4. Model Characteristics Radar Chart
categories = ['Accuracy', 'F1-Score', 'Speed', 'Interpretability', 'Robustness']
convnext_scores = [90.93, 90.34, 85, 30, 80]
sc_convnext_scores = [88.89, 88.85, 85, 30, 90]
protopnet_scores = [70.07, 69.72, 60, 95, 70]

# Number of variables
N = len(categories)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialize the spider plot
ax4 = plt.subplot(2, 2, 4, projection='polar')

# Plot ConvNeXt
values = convnext_scores + convnext_scores[:1]
ax4.plot(angles, values, linewidth=2, linestyle='solid', label='ConvNeXt', color='#2E86AB')
ax4.fill(angles, values, alpha=0.25, color='#2E86AB')

# Plot SC-ConvNeXt
values = sc_convnext_scores + sc_convnext_scores[:1]
ax4.plot(angles, values, linewidth=2, linestyle='solid', label='SC-ConvNeXt', color='#A23B72')
ax4.fill(angles, values, alpha=0.25, color='#A23B72')

# Plot ProtoPNet
values = protopnet_scores + protopnet_scores[:1]
ax4.plot(angles, values, linewidth=2, linestyle='solid', label='ProtoPNet', color='#F18F01')
ax4.fill(angles, values, alpha=0.25, color='#F18F01')

# Fix axis to go in the right order and start at 12 o'clock
ax4.set_theta_offset(np.pi / 2)
ax4.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(categories)

# Add legend
ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
ax4.set_title('Model Characteristics Comparison', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== MODEL PERFORMANCE SUMMARY ===")
print(f"{'Model':<15} {'Accuracy':<10} {'Macro F1':<10} {'Weighted F1':<12}")
print("-" * 50)
for i, model in enumerate(models):
    print(f"{model:<15} {accuracies[i]:<10.2f} {macro_f1[i]:<10.2f} {weighted_f1[i]:<12.2f}")

print("\n=== KEY INSIGHTS ===")
print("1. ConvNeXt achieves the best overall performance (90.93% accuracy)")
print("2. SC-ConvNeXt provides good balance of performance and robustness")
print("3. ProtoPNet offers high interpretability but lower accuracy")
print("4. All models struggle with tan_spot and leaf_blight detection")
print("5. Army_worm and yellow_rust are detected with near-perfect accuracy") 