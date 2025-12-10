import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Model names and their test accuracy at epoch 20 (from classification reports)
model_names = [
    "ConvNeXt",
    "YOLOv9 + EfficientNet",
    "ProtoPNet",
    "Hybrid V2",
    "Hybrid CNN-ViT"
]

accuracies = [
    91.47,  # ConvNeXt
    89.52,  # YOLOv9 + EfficientNet
    69.98,  # ProtoPNet
    89.70,  # Hybrid V2
    90.94   # Hybrid CNN-ViT
]

# Create the bar chart
plt.figure(figsize=(12, 7))
bars = plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], 
               edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on top of each bar
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Model Name', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy %', fontsize=14, fontweight='bold')
plt.title('Model Accuracy Comparison (Epoch 20)', fontsize=16, fontweight='bold', pad=20)
plt.ylim(0, max(accuracies) * 1.15)  # Add some padding at the top
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(rotation=15, ha='right', fontsize=11)
plt.tight_layout()

# Save the chart
output_file = '5_models_accuracy_comparison_epoch20.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Chart saved as: {output_file}")

# Also save as PDF
output_file_pdf = '5_models_accuracy_comparison_epoch20.pdf'
plt.savefig(output_file_pdf, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Chart saved as: {output_file_pdf}")

plt.close()

print("\n" + "="*60)
print("Summary:")
for name, acc in zip(model_names, accuracies):
    print(f"  - {name}: {acc:.2f}%")
print("="*60)
