import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pandas as pd
from datetime import datetime

# Create output directory
EPOCH_REPORT_DIR = 'epoch_training_analysis'
if not os.path.exists(EPOCH_REPORT_DIR):
    os.makedirs(EPOCH_REPORT_DIR)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🚀 Generating epoch-by-epoch training analysis...")

def save_chart(filename, dpi=300):
    """Helper function to save charts"""
    plt.savefig(f'{EPOCH_REPORT_DIR}/{filename}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{EPOCH_REPORT_DIR}/{filename}.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

# Extract epoch data from training logs
# ConvNeXt 10 epochs
convnext_10_train_acc = [63.49, 86.84, 90.77, 91.80, 93.55, 94.66, 95.65, 96.60, 97.02, 96.76]
convnext_10_val_acc = [85.41, 89.50, 89.86, 91.46, 90.75, 91.81, 91.64, 90.21, 90.93, 91.10]
convnext_10_train_loss = [1.2953, 0.4371, 0.3132, 0.2482, 0.1959, 0.1649, 0.1391, 0.1155, 0.1052, 0.1049]
convnext_10_val_loss = [0.5372, 0.3226, 0.3040, 0.2810, 0.2871, 0.2947, 0.2584, 0.2921, 0.2872, 0.2567]

# ConvNeXt 20 epochs
convnext_20_train_acc = [50.74, 81.15, 85.92, 90.42, 90.19, 91.00, 94.66, 94.62, 94.96, 95.65, 96.22, 95.99, 96.49, 97.83, 97.75]
convnext_20_val_acc = [79.00, 83.45, 86.48, 86.48, 87.37, 90.21, 88.26, 87.19, 88.61, 91.46, 90.04, 88.97, 88.43, 90.21, 90.04]
convnext_20_train_loss = [1.6102, 0.6029, 0.4480, 0.3283, 0.3166, 0.2785, 0.1937, 0.1663, 0.1645, 0.1414, 0.1282, 0.1238, 0.1040, 0.0816, 0.0718]
convnext_20_val_loss = [0.6517, 0.4558, 0.3596, 0.3875, 0.3481, 0.3150, 0.3117, 0.4650, 0.3354, 0.2892, 0.3226, 0.3828, 0.3978, 0.3084, 0.3175]

# SC-ConvNeXt 10 epochs
sc_convnext_10_train_acc = [66.08, 87.79, 91.30, 94.58, 94.39, 96.49, 96.49, 97.37, 97.98, 97.90]
sc_convnext_10_val_acc = [84.70, 87.01, 88.97, 89.68, 90.39, 90.21, 91.64, 90.21, 92.53, 91.46]
sc_convnext_10_train_loss = [0.8670, 0.2465, 0.1511, 0.0858, 0.0861, 0.0528, 0.0516, 0.0354, 0.0286, 0.0241]
sc_convnext_10_val_loss = [0.3087, 0.2034, 0.1672, 0.1525, 0.1423, 0.1301, 0.1346, 0.1402, 0.1467, 0.1483]

# SC-ConvNeXt 20 epochs
sc_convnext_20_train_acc = [66.01, 88.78, 91.42, 93.86, 94.93, 96.22, 96.22, 96.34, 97.75, 98.28, 98.32, 98.32, 98.32, 98.82, 99.24, 99.08, 98.93, 99.20, 99.08, 99.24]
sc_convnext_20_val_acc = [80.60, 87.54, 89.15, 91.28, 92.70, 91.81, 90.75, 92.35, 91.28, 91.99, 91.10, 91.10, 91.10, 92.70, 91.99, 91.10, 92.70, 92.17, 92.70, 93.06]
sc_convnext_20_train_loss = [0.8592, 0.2237, 0.1461, 0.1041, 0.0715, 0.0624, 0.0498, 0.0467, 0.0326, 0.0253, 0.0235, 0.0235, 0.0235, 0.0127, 0.0089, 0.0100, 0.0115, 0.0091, 0.0084, 0.0081]
sc_convnext_20_val_loss = [0.3571, 0.2052, 0.1706, 0.1376, 0.1145, 0.1268, 0.1291, 0.1082, 0.1246, 0.1321, 0.1379, 0.1379, 0.1379, 0.1293, 0.1216, 0.1222, 0.1261, 0.1200, 0.1254, 0.1168]

# Hybrid CNN-ViT 10 epochs
hybrid_cnn_vit_10_train_acc = [66.16, 83.52, 88.02, 90.61, 91.84, 92.75, 93.74, 95.31, 95.80, 96.68]
hybrid_cnn_vit_10_val_acc = [82.56, 87.37, 87.19, 88.08, 89.50, 88.43, 91.46, 88.43, 89.86, 90.75]
hybrid_cnn_vit_10_train_loss = [1.0443, 0.4909, 0.3504, 0.2815, 0.2472, 0.2212, 0.1731, 0.1511, 0.1217, 0.0985]
hybrid_cnn_vit_10_val_loss = [0.5143, 0.3950, 0.3663, 0.3405, 0.3333, 0.3545, 0.3013, 0.3313, 0.3567, 0.3161]

# Hybrid CNN-ViT 20 epochs
hybrid_cnn_vit_20_train_acc = [70.47, 84.97, 89.70, 91.26, 92.29, 93.82, 94.96, 95.19, 97.02, 98.02, 98.51, 98.51, 98.63, 98.63, 98.89, 99.08, 98.63]
hybrid_cnn_vit_20_val_acc = [81.67, 82.56, 85.23, 88.97, 91.46, 89.32, 90.39, 89.68, 91.64, 91.46, 91.99, 92.17, 91.81, 91.81, 91.28, 91.46, 91.46]
hybrid_cnn_vit_20_train_loss = [0.9060, 0.4428, 0.3158, 0.2500, 0.2246, 0.1763, 0.1542, 0.1431, 0.0833, 0.0579, 0.0504, 0.0469, 0.0375, 0.0424, 0.0358, 0.0271, 0.0394]
hybrid_cnn_vit_20_val_loss = [0.4977, 0.4836, 0.4312, 0.3131, 0.2890, 0.3450, 0.3723, 0.3272, 0.2642, 0.2593, 0.2567, 0.2534, 0.2660, 0.2658, 0.2838, 0.2819, 0.2829]

# Hybrid V2 10 epochs
hybrid_v2_10_train_acc = [61.88, 78.79, 83.52, 86.23, 85.77, 88.82, 88.63, 90.12, 90.81, 92.37]
hybrid_v2_10_val_acc = [76.69, 83.10, 87.37, 83.81, 86.65, 86.83, 88.43, 87.90, 90.75, 90.04]
hybrid_v2_10_train_loss = [1.2128, 0.6601, 0.5099, 0.4375, 0.4411, 0.3513, 0.3605, 0.2943, 0.2775, 0.2369]
hybrid_v2_10_val_loss = [0.7357, 0.5319, 0.4102, 0.4347, 0.4086, 0.3467, 0.3884, 0.3421, 0.3012, 0.3167]

# Hybrid V2 20 epochs (stopped at epoch 19 due to early stopping)
hybrid_v2_20_train_acc = [66.23, 86.23, 90.12, 91.42, 91.91, 93.29, 94.16, 94.12, 95.15, 95.19, 96.49, 97.67, 98.44, 98.86, 98.36, 98.82, 98.31, 98.82, 98.20]
hybrid_v2_20_val_acc = [83.81, 89.15, 88.08, 88.61, 90.21, 89.50, 88.79, 90.39, 91.10, 91.10, 92.35, 92.35, 92.35, 92.88, 92.70, 92.53, 92.35, 92.35, 92.35]
hybrid_v2_20_train_loss = [1.0472, 0.4059, 0.3036, 0.2541, 0.2333, 0.2010, 0.1704, 0.1772, 0.1274, 0.1366, 0.1062, 0.0699, 0.0457, 0.0336, 0.0444, 0.0294, 0.0318, 0.0278, 0.0386]
hybrid_v2_20_val_loss = [0.4712, 0.3103, 0.4062, 0.3721, 0.2979, 0.3133, 0.3236, 0.2529, 0.3066, 0.3089, 0.3173, 0.2792, 0.2652, 0.2572, 0.2600, 0.2622, 0.2648, 0.2651, 0.2654]

# ProtoPNet 10 epochs
protopnet_10_train_acc = [8.74, 10.68, 14.57, 18.89, 24.69, 29.45, 36.74, 41.43, 46.89, 53.41]
protopnet_10_val_acc = [7.83, 15.12, 16.73, 21.00, 31.49, 32.38, 37.90, 43.77, 50.00, 52.85]
protopnet_10_train_loss = [299.9309, 16.2807, 3.2429, 2.9632, 2.7543, 2.7381, 2.3458, 2.1862, 1.9319, 1.8158]
protopnet_10_val_loss = [60.7292, 4.1492, 3.9562, 3.1752, 2.6265, 2.5972, 2.3018, 2.3123, 2.2351, 2.0765]

# ProtoPNet 20 epochs
protopnet_20_train_acc = [8.70, 11.71, 14.92, 16.06, 21.75, 26.06, 31.36, 33.88, 43.19, 46.81, 48.65, 52.80, 55.55, 58.49, 61.12, 63.49, 64.75, 65.62, 69.74, 73.18]
protopnet_20_val_acc = [7.83, 16.01, 20.11, 17.97, 27.05, 29.89, 31.14, 36.12, 42.53, 41.64, 49.64, 56.76, 60.32, 55.52, 60.68, 58.01, 69.57, 71.89, 71.89, 68.33]
protopnet_20_train_loss = [298.6532, 15.8734, 3.4407, 3.6293, 3.0630, 3.0099, 2.6189, 2.4538, 2.0877, 2.0863, 1.8914, 1.7623, 1.7848, 1.6101, 1.4767, 1.4257, 1.4057, 1.3793, 1.2388, 0.9699]
protopnet_20_val_loss = [61.0047, 5.3483, 4.6805, 4.0803, 3.5412, 3.4432, 3.1155, 2.6300, 2.3383, 2.3809, 1.7861, 1.6161, 1.5000, 1.6529, 1.3413, 1.5591, 1.1747, 1.0082, 1.0726, 1.3808]

# YOLOv9+EfficientNet 20 epochs
yolo_efficientnet_20_train_acc = [21.94, 37.81, 51.32, 61.81, 70.55, 76.12, 79.93, 81.76, 83.44, 83.37, 86.61, 87.71, 89.36, 89.58, 90.61, 91.26, 92.03, 92.98]
yolo_efficientnet_20_val_acc = [37.90, 50.00, 58.72, 76.33, 82.38, 84.88, 85.77, 86.48, 87.90, 88.79, 89.68, 89.50, 88.43, 89.32, 90.04, 90.21, 90.39, 90.21]
yolo_efficientnet_20_train_loss = [2.4073, 1.8356, 1.3731, 1.1118, 0.8932, 0.7385, 0.6289, 0.5544, 0.4903, 0.4836, 0.3798, 0.3562, 0.3126, 0.2952, 0.2796, 0.2555, 0.2314, 0.2076]
yolo_efficientnet_20_val_loss = [2.0078, 1.3487, 1.0428, 0.7963, 0.5925, 0.4909, 0.4221, 0.3713, 0.3513, 0.3508, 0.3155, 0.3414, 0.3595, 0.3408, 0.3278, 0.3255, 0.3112, 0.3335]

print("✅ Extracted epoch-by-epoch data from training logs")

# 1. Overall Performance Comparison: 10 vs 20 Epochs
plt.figure(figsize=(15, 10))
models = ['ConvNeXt', 'SC-ConvNeXt', 'Hybrid CNN-ViT', 'Hybrid V2', 'ProtoPNet']
epochs_10_final = [90.41, 88.10, 88.45, 87.21, 56.13]
epochs_20_final = [91.47, 91.47, 91.65, 92.35, 69.98]

x = np.arange(len(models))
width = 0.35

bars1 = plt.bar(x - width/2, epochs_10_final, width, label='10 Epochs', alpha=0.8, color='#FF6B6B')
bars2 = plt.bar(x + width/2, epochs_20_final, width, label='20 Epochs', alpha=0.8, color='#4ECDC4')

plt.xlabel('Models', fontsize=14)
plt.ylabel('Final Test Accuracy (%)', fontsize=14)
plt.title('Model Performance: 10 vs 20 Epochs Training', fontsize=18, fontweight='bold')
plt.xticks(x, models, rotation=45)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Add improvement annotations
for i, (acc10, acc20) in enumerate(zip(epochs_10_final, epochs_20_final)):
    improvement = acc20 - acc10
    plt.annotate(f'+{improvement:.1f}%', 
                xy=(i + width/2, acc20), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
save_chart('01_10_vs_20_epochs_comparison')
print("✅ Chart 1: 10 vs 20 Epochs Comparison")

# 2. Training Progress Over Time - All Models
plt.figure(figsize=(20, 12))

# Subplot 1: ConvNeXt
plt.subplot(2, 3, 1)
epochs_10 = list(range(1, 11))
epochs_20 = list(range(1, 16))  # Early stopping at epoch 15
plt.plot(epochs_10, convnext_10_val_acc, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20, convnext_20_val_acc, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('ConvNeXt Training Progress', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(75, 95)

# Subplot 2: SC-ConvNeXt
plt.subplot(2, 3, 2)
epochs_20_full = list(range(1, 21))
plt.plot(epochs_10, sc_convnext_10_val_acc, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20_full, sc_convnext_20_val_acc, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('SC-ConvNeXt Training Progress', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(75, 95)

# Subplot 3: Hybrid CNN-ViT
plt.subplot(2, 3, 3)
epochs_20_early = list(range(1, 18))  # Early stopping at epoch 17
plt.plot(epochs_10, hybrid_cnn_vit_10_val_acc, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20_early, hybrid_cnn_vit_20_val_acc, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('Hybrid CNN-ViT Training Progress', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(75, 95)

# Subplot 4: Hybrid V2
plt.subplot(2, 3, 4)
epochs_20_v2 = list(range(1, 20))  # Early stopping at epoch 19
plt.plot(epochs_10, hybrid_v2_10_val_acc, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20_v2, hybrid_v2_20_val_acc, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('Hybrid V2 Training Progress', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(75, 95)

# Subplot 5: ProtoPNet
plt.subplot(2, 3, 5)
epochs_20_proto = list(range(1, 21))
plt.plot(epochs_10, protopnet_10_val_acc, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20_proto, protopnet_20_val_acc, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('ProtoPNet Training Progress', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 75)

# Subplot 6: YOLOv9+EfficientNet (20 epochs only)
plt.subplot(2, 3, 6)
epochs_20_yolo = list(range(1, 19))  # Stopped at epoch 18
plt.plot(epochs_20_yolo, yolo_efficientnet_20_val_acc, 's-', label='20 Epochs', linewidth=2, markersize=6, color='purple')
plt.title('YOLOv9+EfficientNet Training Progress', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(30, 95)

plt.suptitle('Training Progress Comparison: All Models', fontsize=20, fontweight='bold')
plt.tight_layout()
save_chart('02_training_progress_all_models')
print("✅ Chart 2: Training Progress All Models")

# 3. Loss Curves Comparison
plt.figure(figsize=(20, 12))

# Subplot 1: ConvNeXt Loss
plt.subplot(2, 3, 1)
plt.plot(epochs_10, convnext_10_val_loss, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20, convnext_20_val_loss, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('ConvNeXt Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Subplot 2: SC-ConvNeXt Loss
plt.subplot(2, 3, 2)
plt.plot(epochs_10, sc_convnext_10_val_loss, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20_full, sc_convnext_20_val_loss, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('SC-ConvNeXt Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Subplot 3: Hybrid CNN-ViT Loss
plt.subplot(2, 3, 3)
plt.plot(epochs_10, hybrid_cnn_vit_10_val_loss, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20_early, hybrid_cnn_vit_20_val_loss, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('Hybrid CNN-ViT Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Subplot 4: Hybrid V2 Loss
plt.subplot(2, 3, 4)
plt.plot(epochs_10, hybrid_v2_10_val_loss, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20_v2, hybrid_v2_20_val_loss, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('Hybrid V2 Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Subplot 5: ProtoPNet Loss
plt.subplot(2, 3, 5)
plt.plot(epochs_10, protopnet_10_val_loss, 'o-', label='10 Epochs', linewidth=2, markersize=6)
plt.plot(epochs_20_proto, protopnet_20_val_loss, 's-', label='20 Epochs', linewidth=2, markersize=6)
plt.title('ProtoPNet Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Subplot 6: YOLOv9+EfficientNet Loss
plt.subplot(2, 3, 6)
plt.plot(epochs_20_yolo, yolo_efficientnet_20_val_loss, 's-', label='20 Epochs', linewidth=2, markersize=6, color='purple')
plt.title('YOLOv9+EfficientNet Validation Loss', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.suptitle('Validation Loss Curves: All Models', fontsize=20, fontweight='bold')
plt.tight_layout()
save_chart('03_validation_loss_curves')
print("✅ Chart 3: Validation Loss Curves")

# 4. Individual Model Detailed Analysis
def create_detailed_model_plot(model_name, train_acc_10, val_acc_10, train_loss_10, val_loss_10,
                              train_acc_20, val_acc_20, train_loss_20, val_loss_20,
                              epochs_10, epochs_20, filename):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training Accuracy
    ax1.plot(epochs_10, train_acc_10, 'o-', label='10 Epochs', linewidth=2, markersize=6)
    ax1.plot(epochs_20, train_acc_20, 's-', label='20 Epochs', linewidth=2, markersize=6)
    ax1.set_title(f'{model_name} - Training Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax2.plot(epochs_10, val_acc_10, 'o-', label='10 Epochs', linewidth=2, markersize=6)
    ax2.plot(epochs_20, val_acc_20, 's-', label='20 Epochs', linewidth=2, markersize=6)
    ax2.set_title(f'{model_name} - Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training Loss
    ax3.plot(epochs_10, train_loss_10, 'o-', label='10 Epochs', linewidth=2, markersize=6)
    ax3.plot(epochs_20, train_loss_20, 's-', label='20 Epochs', linewidth=2, markersize=6)
    ax3.set_title(f'{model_name} - Training Loss', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Validation Loss
    ax4.plot(epochs_10, val_loss_10, 'o-', label='10 Epochs', linewidth=2, markersize=6)
    ax4.plot(epochs_20, val_loss_20, 's-', label='20 Epochs', linewidth=2, markersize=6)
    ax4.set_title(f'{model_name} - Validation Loss', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.suptitle(f'{model_name} - Detailed Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_chart(filename)

# Create detailed plots for each model
create_detailed_model_plot('ConvNeXt', convnext_10_train_acc, convnext_10_val_acc, 
                          convnext_10_train_loss, convnext_10_val_loss,
                          convnext_20_train_acc, convnext_20_val_acc,
                          convnext_20_train_loss, convnext_20_val_loss,
                          epochs_10, epochs_20, '04_convnext_detailed')
print("✅ Chart 4: ConvNeXt Detailed Analysis")

create_detailed_model_plot('SC-ConvNeXt', sc_convnext_10_train_acc, sc_convnext_10_val_acc,
                          sc_convnext_10_train_loss, sc_convnext_10_val_loss,
                          sc_convnext_20_train_acc, sc_convnext_20_val_acc,
                          sc_convnext_20_train_loss, sc_convnext_20_val_loss,
                          epochs_10, epochs_20_full, '05_sc_convnext_detailed')
print("✅ Chart 5: SC-ConvNeXt Detailed Analysis")

create_detailed_model_plot('Hybrid CNN-ViT', hybrid_cnn_vit_10_train_acc, hybrid_cnn_vit_10_val_acc,
                          hybrid_cnn_vit_10_train_loss, hybrid_cnn_vit_10_val_loss,
                          hybrid_cnn_vit_20_train_acc, hybrid_cnn_vit_20_val_acc,
                          hybrid_cnn_vit_20_train_loss, hybrid_cnn_vit_20_val_loss,
                          epochs_10, epochs_20_early, '06_hybrid_cnn_vit_detailed')
print("✅ Chart 6: Hybrid CNN-ViT Detailed Analysis")

create_detailed_model_plot('Hybrid V2', hybrid_v2_10_train_acc, hybrid_v2_10_val_acc,
                          hybrid_v2_10_train_loss, hybrid_v2_10_val_loss,
                          hybrid_v2_20_train_acc, hybrid_v2_20_val_acc,
                          hybrid_v2_20_train_loss, hybrid_v2_20_val_loss,
                          epochs_10, epochs_20_v2, '07_hybrid_v2_detailed')
print("✅ Chart 7: Hybrid V2 Detailed Analysis")

create_detailed_model_plot('ProtoPNet', protopnet_10_train_acc, protopnet_10_val_acc,
                          protopnet_10_train_loss, protopnet_10_val_loss,
                          protopnet_20_train_acc, protopnet_20_val_acc,
                          protopnet_20_train_loss, protopnet_20_val_loss,
                          epochs_10, epochs_20_proto, '08_protopnet_detailed')
print("✅ Chart 8: ProtoPNet Detailed Analysis")

# 5. Performance Improvement Analysis
plt.figure(figsize=(15, 10))
improvements = [acc20 - acc10 for acc10, acc20 in zip(epochs_10_final, epochs_20_final)]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD']

bars = plt.bar(models, improvements, color=colors, alpha=0.8)
plt.title('Performance Improvement: 20 vs 10 Epochs', fontsize=18, fontweight='bold')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Accuracy Improvement (%)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'+{improvements[i]:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
save_chart('09_performance_improvement')
print("✅ Chart 9: Performance Improvement Analysis")

# 6. Early Stopping Analysis
plt.figure(figsize=(12, 8))
models_with_early_stopping = ['ConvNeXt', 'Hybrid CNN-ViT', 'Hybrid V2']
early_stopping_epochs = [15, 17, 19]
final_accuracies = [91.47, 91.65, 92.35]

bars = plt.bar(models_with_early_stopping, early_stopping_epochs, 
               color=['#FF6B6B', '#4ECDC4', '#96CEB4'], alpha=0.8)
plt.title('Early Stopping Analysis', fontsize=18, fontweight='bold')
plt.xlabel('Models', fontsize=14)
plt.ylabel('Epoch When Stopped', fontsize=14)
plt.xticks(rotation=45)

# Add accuracy annotations
for i, (bar, acc) in enumerate(zip(bars, final_accuracies)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
save_chart('10_early_stopping_analysis')
print("✅ Chart 10: Early Stopping Analysis")

# 7. Comprehensive Dashboard
fig = plt.figure(figsize=(24, 16))

# Main title
fig.suptitle('Wheat Disease Detection: Comprehensive Training Analysis Dashboard', 
             fontsize=24, fontweight='bold', y=0.98)

# Subplot 1: Performance Comparison
ax1 = plt.subplot(3, 4, 1)
bars1 = ax1.bar(x - width/2, epochs_10_final, width, label='10 Epochs', alpha=0.8, color='#FF6B6B')
bars2 = ax1.bar(x + width/2, epochs_20_final, width, label='20 Epochs', alpha=0.8, color='#4ECDC4')
ax1.set_title('Final Accuracy Comparison', fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Improvement Analysis
ax2 = plt.subplot(3, 4, 2)
bars = ax2.bar(models, improvements, color=colors, alpha=0.8)
ax2.set_title('Performance Improvement', fontweight='bold')
ax2.set_ylabel('Improvement (%)')
ax2.set_xticklabels(models, rotation=45)
ax2.grid(True, alpha=0.3)

# Subplot 3: Early Stopping
ax3 = plt.subplot(3, 4, 3)
bars = ax3.bar(models_with_early_stopping, early_stopping_epochs, 
               color=['#FF6B6B', '#4ECDC4', '#96CEB4'], alpha=0.8)
ax3.set_title('Early Stopping Epochs', fontweight='bold')
ax3.set_ylabel('Epoch')
ax3.set_xticklabels(models_with_early_stopping, rotation=45)
ax3.grid(True, alpha=0.3)

# Subplot 4: Best Performers
ax4 = plt.subplot(3, 4, 4)
best_models = ['Hybrid V2', 'Hybrid CNN-ViT', 'ConvNeXt', 'SC-ConvNeXt']
best_accuracies = [92.35, 91.65, 91.47, 91.47]
bars = ax4.bar(best_models, best_accuracies, color=['#96CEB4', '#4ECDC4', '#FF6B6B', '#45B7D1'], alpha=0.8)
ax4.set_title('Top Performers (20 Epochs)', fontweight='bold')
ax4.set_ylabel('Accuracy (%)')
ax4.set_xticklabels(best_models, rotation=45)
ax4.grid(True, alpha=0.3)

# Subplot 5-8: Individual model progress (simplified)
models_data = [
    ('ConvNeXt', convnext_10_val_acc, convnext_20_val_acc, epochs_10, epochs_20),
    ('SC-ConvNeXt', sc_convnext_10_val_acc, sc_convnext_20_val_acc, epochs_10, epochs_20_full),
    ('Hybrid CNN-ViT', hybrid_cnn_vit_10_val_acc, hybrid_cnn_vit_20_val_acc, epochs_10, epochs_20_early),
    ('Hybrid V2', hybrid_v2_10_val_acc, hybrid_v2_20_val_acc, epochs_10, epochs_20_v2)
]

for i, (model_name, val_acc_10, val_acc_20, ep_10, ep_20) in enumerate(models_data):
    ax = plt.subplot(3, 4, 5 + i)
    ax.plot(ep_10, val_acc_10, 'o-', label='10 Epochs', linewidth=2, markersize=4)
    ax.plot(ep_20, val_acc_20, 's-', label='20 Epochs', linewidth=2, markersize=4)
    ax.set_title(f'{model_name} Progress', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Subplot 9-12: Loss curves (simplified)
for i, (model_name, val_loss_10, val_loss_20, ep_10, ep_20) in enumerate([
    ('ConvNeXt', convnext_10_val_loss, convnext_20_val_loss, epochs_10, epochs_20),
    ('SC-ConvNeXt', sc_convnext_10_val_loss, sc_convnext_20_val_loss, epochs_10, epochs_20_full),
    ('Hybrid CNN-ViT', hybrid_cnn_vit_10_val_loss, hybrid_cnn_vit_20_val_loss, epochs_10, epochs_20_early),
    ('Hybrid V2', hybrid_v2_10_val_loss, hybrid_v2_20_val_loss, epochs_10, epochs_20_v2)
]):
    ax = plt.subplot(3, 4, 9 + i)
    ax.plot(ep_10, val_loss_10, 'o-', label='10 Epochs', linewidth=2, markersize=4)
    ax.plot(ep_20, val_loss_20, 's-', label='20 Epochs', linewidth=2, markersize=4)
    ax.set_title(f'{model_name} Loss', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

plt.tight_layout()
save_chart('11_comprehensive_dashboard')
print("✅ Chart 11: Comprehensive Dashboard")

# Generate summary report
report_content = f"""
# Epoch-by-Epoch Training Analysis Report

## Overview
This report analyzes the training progress of 6 different models across 10 and 20 epochs for wheat disease detection.

## Key Findings

### Performance Comparison (10 vs 20 Epochs)
- **Hybrid V2**: 87.21% -> 92.35% (+5.14% improvement)
- **SC-ConvNeXt**: 88.10% -> 91.47% (+3.37% improvement)  
- **Hybrid CNN-ViT**: 88.45% -> 91.65% (+3.20% improvement)
- **ConvNeXt**: 90.41% -> 91.47% (+1.06% improvement)
- **ProtoPNet**: 56.13% -> 69.98% (+13.85% improvement)

### Early Stopping Analysis
- **ConvNeXt**: Stopped at epoch 15 (91.47% accuracy)
- **Hybrid CNN-ViT**: Stopped at epoch 17 (91.65% accuracy)
- **Hybrid V2**: Stopped at epoch 19 (92.35% accuracy)

### Training Insights
1. **Hybrid V2** achieved the best overall performance (92.35%)
2. **ProtoPNet** showed the most improvement with longer training
3. **Early stopping** was effective for preventing overfitting
4. **20 epochs** generally provided better performance than 10 epochs

## Generated Visualizations
1. 10 vs 20 Epochs Comparison
2. Training Progress All Models
3. Validation Loss Curves
4-8. Detailed Analysis for Each Model
9. Performance Improvement Analysis
10. Early Stopping Analysis
11. Comprehensive Dashboard

All charts are saved in both PNG and PDF formats in the '{EPOCH_REPORT_DIR}/' directory.
"""

# Save the report
with open(f'{EPOCH_REPORT_DIR}/epoch_analysis_report.md', 'w') as f:
    f.write(report_content)

# Create summary CSV
summary_data = {
    'Model': models + ['YOLOv9+EfficientNet'],
    '10_Epochs_Accuracy': epochs_10_final + [None],
    '20_Epochs_Accuracy': epochs_20_final + [89.52],
    'Improvement': improvements + [None],
    'Early_Stopping_Epoch': [15, None, 17, 19, None, None],
    'Best_Accuracy': [91.47, 91.47, 91.65, 92.35, 69.98, 89.52]
}

df = pd.DataFrame(summary_data)
df.to_csv(f'{EPOCH_REPORT_DIR}/epoch_analysis_summary.csv', index=False)

print("\n🎉 Epoch-by-epoch training analysis completed successfully!")
print(f"📁 Output directory: {EPOCH_REPORT_DIR}/")
print(f"📊 Total charts generated: 11")
print(f"📄 Report files:")
print(f"   - epoch_analysis_report.md (Detailed analysis)")
print(f"   - epoch_analysis_summary.csv (Data summary)")
print(f"   - 11 chart files (PNG + PDF formats)")
print(f"\n💡 All files are saved in the '{EPOCH_REPORT_DIR}/' folder!")
