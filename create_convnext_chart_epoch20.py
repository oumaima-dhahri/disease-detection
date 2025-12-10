import matplotlib.pyplot as plt
import re
import os

# Data from train convnext.txt
log_content = """
Epoch 1/20 | Train Loss: 1.6102 Acc: 0.5074 | Val Loss: 0.6517 Acc: 0.7900
Epoch 2/20 | Train Loss: 0.6029 Acc: 0.8115 | Val Loss: 0.4558 Acc: 0.8345
Epoch 3/20 | Train Loss: 0.4480 Acc: 0.8592 | Val Loss: 0.3596 Acc: 0.8648
Epoch 4/20 | Train Loss: 0.3283 Acc: 0.9042 | Val Loss: 0.3875 Acc: 0.8648
Epoch 5/20 | Train Loss: 0.3166 Acc: 0.9019 | Val Loss: 0.3481 Acc: 0.8737
Epoch 6/20 | Train Loss: 0.2785 Acc: 0.9100 | Val Loss: 0.3150 Acc: 0.9021
Epoch 7/20 | Train Loss: 0.1937 Acc: 0.9466 | Val Loss: 0.3117 Acc: 0.8826
Epoch 8/20 | Train Loss: 0.1663 Acc: 0.9462 | Val Loss: 0.4650 Acc: 0.8719
Epoch 9/20 | Train Loss: 0.1645 Acc: 0.9496 | Val Loss: 0.3354 Acc: 0.8861
Epoch 10/20 | Train Loss: 0.1414 Acc: 0.9565 | Val Loss: 0.2892 Acc: 0.9146
Epoch 11/20 | Train Loss: 0.1282 Acc: 0.9622 | Val Loss: 0.3226 Acc: 0.9004
Epoch 12/20 | Train Loss: 0.1238 Acc: 0.9599 | Val Loss: 0.3828 Acc: 0.8897
Epoch 13/20 | Train Loss: 0.1040 Acc: 0.9649 | Val Loss: 0.3978 Acc: 0.8843
Epoch 14/20 | Train Loss: 0.0816 Acc: 0.9783 | Val Loss: 0.3084 Acc: 0.9021
Epoch 15/20 | Train Loss: 0.0718 Acc: 0.9775 | Val Loss: 0.3175 Acc: 0.9004
"""

epochs = []
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# Parse the log content
for line in log_content.strip().split('\n'):
    match = re.search(r'Epoch (\d+)/20 \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)', line)
    if match:
        epochs.append(int(match.group(1)))
        train_loss.append(float(match.group(2)))
        train_acc.append(float(match.group(3)))
        val_loss.append(float(match.group(4)))
        val_acc.append(float(match.group(5)))

# Create the plots
plt.figure(figsize=(15, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o', color='blue')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o', color='orange')
plt.title('ConvNeXt (No SC) Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss', marker='o', color='blue')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='red')
plt.title('ConvNeXt (No SC) Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('convnext_training_curves_epoch20_generated.png')
print("Chart saved as convnext_training_curves_epoch20_generated.png")
