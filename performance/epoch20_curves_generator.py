import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re

def extract_epoch_data_from_log(log_file_path):
    """Extract epoch data from training log files"""
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Different patterns for different model formats
        patterns = [
            # SC-ConvNeXt pattern (with emoji)
            r'Epoch (\d+)/20 \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)',
            # ConvNeXt pattern
            r'Epoch (\d+)/20 \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)',
            # Hybrid CNN-ViT pattern
            r'Epoch (\d+)/20 \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)',
            # YOLOv9 + EfficientNet pattern
            r'Epoch (\d+)/20:\s+Train Loss: ([\d.]+), Train Acc: ([\d.]+)%\s+Val Loss: ([\d.]+), Val Acc: ([\d.]+)%',
            # ProtoPNet pattern
            r'Epoch (\d+)/20\s+Train Loss: ([\d.]+) Acc: ([\d.]+)\s+Val Loss: ([\d.]+) Acc: ([\d.]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    epoch = int(match[0])
                    if 'YOLOv9' in str(log_file_path) or 'yolo9' in str(log_file_path):
                        # YOLOv9 format has percentage values
                        train_loss = float(match[1])
                        train_acc = float(match[2]) / 100.0  # Convert percentage to decimal
                        val_loss = float(match[3])
                        val_acc = float(match[4]) / 100.0
                    else:
                        train_loss = float(match[1])
                        train_acc = float(match[2])
                        val_loss = float(match[3])
                        val_acc = float(match[4])
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                break
                
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        
    return epochs, train_losses, train_accs, val_losses, val_accs

def create_epoch_curves():
    """Create epoch curves for all models"""
    
    # Define model log files
    model_logs = {
        'SC-ConvNeXt': 'epoch20/output trainig/Train sc convnext.txt',
        'ConvNeXt': 'epoch20/output trainig/train convnext.txt',
        'Hybrid CNN-ViT': 'epoch20/output trainig/train hybrid cnn.txt',
        'YOLOv9 + EfficientNet': 'epoch20/output trainig/train yolo9 efficient net b3.txt',
        'ProtoPNet': 'epoch20/output trainig/train protopnet.txt'
    }
    
    # Extract data for all models
    model_data = {}
    for model_name, log_path in model_logs.items():
        print(f"Processing {model_name}...")
        epochs, train_losses, train_accs, val_losses, val_accs = extract_epoch_data_from_log(log_path)
        if epochs:
            model_data[model_name] = {
                'epochs': epochs,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }
            print(f"  Found {len(epochs)} epochs")
        else:
            print(f"  No epoch data found")
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Training Accuracy
    for i, (model_name, data) in enumerate(model_data.items()):
        ax1.plot(data['epochs'], data['train_accs'], 
                label=f"{model_name} (Final: {data['train_accs'][-1]:.3f})", 
                color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4)
    
    ax1.set_title('Training Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Validation Accuracy
    for i, (model_name, data) in enumerate(model_data.items()):
        ax2.plot(data['epochs'], data['val_accs'], 
                label=f"{model_name} (Final: {data['val_accs'][-1]:.3f})", 
                color=colors[i % len(colors)], linewidth=2, marker='s', markersize=4)
    
    ax2.set_title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Training Loss
    for i, (model_name, data) in enumerate(model_data.items()):
        ax3.plot(data['epochs'], data['train_losses'], 
                label=f"{model_name} (Final: {data['train_losses'][-1]:.3f})", 
                color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4)
    
    ax3.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for better visualization
    
    # Plot 4: Validation Loss
    for i, (model_name, data) in enumerate(model_data.items()):
        ax4.plot(data['epochs'], data['val_losses'], 
                label=f"{model_name} (Final: {data['val_losses'][-1]:.3f})", 
                color=colors[i % len(colors)], linewidth=2, marker='s', markersize=4)
    
    ax4.set_title('Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig('epoch20_all_models_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual model curves
    for model_name, data in model_data.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy curves
        ax1.plot(data['epochs'], data['train_accs'], 'b-', label='Training', linewidth=2, marker='o', markersize=4)
        ax1.plot(data['epochs'], data['val_accs'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
        ax1.set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Loss curves
        ax2.plot(data['epochs'], data['train_losses'], 'b-', label='Training', linewidth=2, marker='o', markersize=4)
        ax2.plot(data['epochs'], data['val_losses'], 'r-', label='Validation', linewidth=2, marker='s', markersize=4)
        ax2.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        filename = f'epoch20_{model_name.lower().replace(" ", "_").replace("+", "_")}_curves.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved {filename}")
    
    # Create summary table
    print("\n" + "="*80)
    print("EPOCH 20 TRAINING SUMMARY")
    print("="*80)
    
    summary_data = []
    for model_name, data in model_data.items():
        summary_data.append({
            'Model': model_name,
            'Epochs': len(data['epochs']),
            'Final Train Acc': f"{data['train_accs'][-1]:.4f}",
            'Final Val Acc': f"{data['val_accs'][-1]:.4f}",
            'Final Train Loss': f"{data['train_losses'][-1]:.4f}",
            'Final Val Loss': f"{data['val_losses'][-1]:.4f}",
            'Best Val Acc': f"{max(data['val_accs']):.4f}",
            'Best Epoch': data['epochs'][data['val_accs'].index(max(data['val_accs']))]
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    return model_data

if __name__ == "__main__":
    print("Generating epoch curves for all models...")
    model_data = create_epoch_curves()
    print("\nEpoch curves generated successfully!")
