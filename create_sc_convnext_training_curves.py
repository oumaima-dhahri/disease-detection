import matplotlib.pyplot as plt
import re
import numpy as np

# Read the training log file
log_file = 'epoch20/output trainig/Train sc convnext.txt'

epochs = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []

# Parse the log file
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Match pattern: 🚀 Epoch X/20 | Train Loss: X.XXXX Acc: X.XXXX | Val Loss: X.XXXX Acc: X.XXXX
        match = re.search(r'Epoch (\d+)/20.*Train Loss: ([\d.]+) Acc: ([\d.]+).*Val Loss: ([\d.]+) Acc: ([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            train_acc = float(match.group(3))
            val_loss = float(match.group(4))
            val_acc = float(match.group(5))
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

print(f"Extracted data for {len(epochs)} epochs")
print(f"Epochs: {epochs}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Training and Validation Loss
ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2.5, markersize=8, alpha=0.8)
ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2.5, markersize=8, alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training and Validation Loss - SC-ConvNext', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, max(epochs) + 1)
ax1.set_xticks(epochs)

# Add value annotations for last epoch
ax1.annotate(f'{train_losses[-1]:.4f}', 
             xy=(epochs[-1], train_losses[-1]), 
             xytext=(5, 5), textcoords='offset points',
             fontsize=9, fontweight='bold', color='blue')
ax1.annotate(f'{val_losses[-1]:.4f}', 
             xy=(epochs[-1], val_losses[-1]), 
             xytext=(5, -15), textcoords='offset points',
             fontsize=9, fontweight='bold', color='red')

# Plot 2: Training and Validation Accuracy
ax2.plot(epochs, [acc * 100 for acc in train_accs], 'b-o', label='Training Accuracy', linewidth=2.5, markersize=8, alpha=0.8)
ax2.plot(epochs, [acc * 100 for acc in val_accs], 'r-s', label='Validation Accuracy', linewidth=2.5, markersize=8, alpha=0.8)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training and Validation Accuracy - SC-ConvNext', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='best')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, max(epochs) + 1)
ax2.set_xticks(epochs)
ax2.set_ylim(0, 100)

# Add value annotations for last epoch
ax2.annotate(f'{train_accs[-1]*100:.2f}%', 
             xy=(epochs[-1], train_accs[-1]*100), 
             xytext=(5, 5), textcoords='offset points',
             fontsize=9, fontweight='bold', color='blue')
ax2.annotate(f'{val_accs[-1]*100:.2f}%', 
             xy=(epochs[-1], val_accs[-1]*100), 
             xytext=(5, -15), textcoords='offset points',
             fontsize=9, fontweight='bold', color='red')

plt.tight_layout()

# Save the figure
output_file = 'sc_convnext_training_curves.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Training curves saved as: {output_file}")

# Also save individual plots
fig1, ax1_alone = plt.subplots(figsize=(10, 6))
ax1_alone.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2.5, markersize=8, alpha=0.8)
ax1_alone.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2.5, markersize=8, alpha=0.8)
ax1_alone.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1_alone.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1_alone.set_title('Training and Validation Loss - SC-ConvNext', fontsize=14, fontweight='bold')
ax1_alone.legend(fontsize=11, loc='best')
ax1_alone.grid(True, alpha=0.3, linestyle='--')
ax1_alone.set_xlim(0, max(epochs) + 1)
ax1_alone.set_xticks(epochs)
plt.tight_layout()
plt.savefig('sc_convnext_loss_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Loss curve saved as: sc_convnext_loss_curve.png")
plt.close()

fig2, ax2_alone = plt.subplots(figsize=(10, 6))
ax2_alone.plot(epochs, [acc * 100 for acc in train_accs], 'b-o', label='Training Accuracy', linewidth=2.5, markersize=8, alpha=0.8)
ax2_alone.plot(epochs, [acc * 100 for acc in val_accs], 'r-s', label='Validation Accuracy', linewidth=2.5, markersize=8, alpha=0.8)
ax2_alone.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2_alone.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2_alone.set_title('Training and Validation Accuracy - SC-ConvNext', fontsize=14, fontweight='bold')
ax2_alone.legend(fontsize=11, loc='best')
ax2_alone.grid(True, alpha=0.3, linestyle='--')
ax2_alone.set_xlim(0, max(epochs) + 1)
ax2_alone.set_xticks(epochs)
ax2_alone.set_ylim(0, 100)
plt.tight_layout()
plt.savefig('sc_convnext_accuracy_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Accuracy curve saved as: sc_convnext_accuracy_curve.png")
plt.close()

print("\n" + "="*60)
print("Summary:")
print(f"  - Total epochs trained: {len(epochs)}")
print(f"  - Final training loss: {train_losses[-1]:.4f}")
print(f"  - Final validation loss: {val_losses[-1]:.4f}")
print(f"  - Final training accuracy: {train_accs[-1]*100:.2f}%")
print(f"  - Final validation accuracy: {val_accs[-1]*100:.2f}%")
print("="*60)

