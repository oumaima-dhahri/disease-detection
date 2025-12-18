import matplotlib.pyplot as plt
import re
import numpy as np

# Read the training log file
log_file = 'epoch20/output trainig/hybrid cnn vit.txt'

epochs = []
train_losses = []
val_losses = []

# Parse the log file
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Match pattern: Epoch X/20 | Train Loss: X.XXXX Acc: X.XXXX | Val Loss: X.XXXX Acc: X.XXXX
        match = re.search(r'Epoch (\d+)/20.*Train Loss: ([\d.]+).*Val Loss: ([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

print(f"Extracted data for {len(epochs)} epochs")
print(f"Epochs: {epochs}")

# Create figure for loss curves
fig, ax = plt.subplots(figsize=(12, 7))

# Plot Training and Validation Loss
ax.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2.5, markersize=8, alpha=0.8, markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=2)
ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2.5, markersize=8, alpha=0.8, markerfacecolor='lightcoral', markeredgecolor='red', markeredgewidth=2)

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('Hybrid CNN-ViT: Training and Validation Loss (Epoch 20)', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.set_xlim(0, max(epochs) + 1)
ax.set_xticks(epochs)

# Add value annotations for last epoch
ax.annotate(f'Train: {train_losses[-1]:.4f}', 
             xy=(epochs[-1], train_losses[-1]), 
             xytext=(10, 10), textcoords='offset points',
             fontsize=10, fontweight='bold', color='blue',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
ax.annotate(f'Val: {val_losses[-1]:.4f}', 
             xy=(epochs[-1], val_losses[-1]), 
             xytext=(10, -20), textcoords='offset points',
             fontsize=10, fontweight='bold', color='red',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

# Highlight best validation loss
best_val_loss_idx = np.argmin(val_losses)
best_val_loss_epoch = epochs[best_val_loss_idx]
best_val_loss = val_losses[best_val_loss_idx]
ax.axvline(x=best_val_loss_epoch, color='green', linestyle='--', alpha=0.6, linewidth=2, label=f'Best Val Loss (Epoch {best_val_loss_epoch})')
ax.plot(best_val_loss_epoch, best_val_loss, 'go', markersize=12, markeredgecolor='darkgreen', markeredgewidth=2)
ax.annotate(f'Best: {best_val_loss:.4f}', 
             xy=(best_val_loss_epoch, best_val_loss), 
             xytext=(15, 15), textcoords='offset points',
             fontsize=9, fontweight='bold', color='green',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

# Update legend to include best validation loss
ax.legend(fontsize=12, loc='upper right', framealpha=0.9, shadow=True)

plt.tight_layout()

# Save the figure
output_file = 'hybrid_cnn_vit_loss_curve_epoch20.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Loss curve saved as: {output_file}")

# Print summary
print("\n" + "="*60)
print("Summary:")
print(f"  - Total epochs trained: {len(epochs)}")
print(f"  - Final training loss: {train_losses[-1]:.4f}")
print(f"  - Final validation loss: {val_losses[-1]:.4f}")
print(f"  - Best validation loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})")
print("="*60)

plt.show()












