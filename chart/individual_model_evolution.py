#!/usr/bin/env python3
"""
Individual Model Evolution Analysis
Creates separate plots for each model showing their epoch-by-epoch validation accuracy evolution
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

def extract_validation_metrics(file_path, model_type):
    """Extract validation accuracy for each epoch."""
    epochs = []
    val_accuracies = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if model_type == 'protopnet':
            # ProtoPNet format: Epoch X/10\nTrain Loss: X Acc: X\nVal Loss: X Acc: X
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('Epoch ') and '/10' in line:
                    epoch_match = re.search(r'Epoch (\d+)/10', line)
                    if epoch_match:
                        epoch_num = int(epoch_match.group(1))
                        epochs.append(epoch_num)
                        
                        # Get validation accuracy from next lines
                        if i + 2 < len(lines):
                            val_line = lines[i + 2].strip()
                            val_match = re.search(r'Val Loss: ([\d.]+) Acc: ([\d.]+)', val_line)
                            if val_match:
                                val_accuracies.append(float(val_match.group(2)))
        
        elif model_type == 'yolov9':
            # YOLOv9 format: Epoch X/10:\n  Train Loss: X, Train Acc: X%\n  Val Loss: X, Val Acc: X%
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('Epoch ') and '/10:' in line:
                    epoch_match = re.search(r'Epoch (\d+)/10:', line)
                    if epoch_match:
                        epoch_num = int(epoch_match.group(1))
                        epochs.append(epoch_num)
                        
                        # Get validation accuracy from next lines
                        if i + 2 < len(lines):
                            val_line = lines[i + 2].strip()
                            val_match = re.search(r'Val Loss: ([\d.]+), Val Acc: ([\d.]+)%', val_line)
                            if val_match:
                                val_accuracies.append(float(val_match.group(2)) / 100)
        
        else:
            # Standard format: Epoch X/Y | Train Loss: X Acc: X | Val Loss: X Acc: X
            epoch_pattern = r'Epoch (\d+)/(\d+) \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)'
            matches = re.findall(epoch_pattern, content)
            for match in matches:
                epoch_num, total_epochs, train_loss, train_acc, val_loss, val_acc = match
                epochs.append(int(epoch_num))
                val_accuracies.append(float(val_acc))
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return epochs, val_accuracies

def create_individual_model_plots():
    """Create separate plots for each model's evolution."""
    
    # Define models and their file mappings
    models = {
        'ConvNeXt': ('convnext.txt', 'standard'),
        'Hybrid_CNN_ViT': ('hybride_cnn_vit.txt', 'standard'),
        'Hybrid_V2': ('hybride_v2.txt', 'standard'),
        'ProtoPNet': ('protopnet .txt', 'protopnet'),
        'SC_ConvNeXt': ('sc-convnext.txt', 'standard'),
        'YOLOv9_EfficientNet': ('yolov9 and efficient b3.txt', 'yolov9')
    }
    
    # Define colors and styles for each model
    model_styles = {
        'ConvNeXt': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},
        'Hybrid_CNN_ViT': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
        'Hybrid_V2': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-'},
        'ProtoPNet': {'color': '#d62728', 'marker': 'D', 'linestyle': '-'},
        'SC_ConvNeXt': {'color': '#9467bd', 'marker': 'v', 'linestyle': '-'},
        'YOLOv9_EfficientNet': {'color': '#8c564b', 'marker': 'p', 'linestyle': '-'}
    }
    
    # Load data for all models
    model_data = {}
    
    print("Loading validation data for all models...")
    for model_name, (filename, model_type) in models.items():
        file_path = os.path.join("epoch10/output trainig", filename)
        if os.path.exists(file_path):
            epochs, val_acc = extract_validation_metrics(file_path, model_type)
            if epochs and val_acc:
                model_data[model_name] = {'epochs': epochs, 'val_acc': val_acc}
                print(f"✓ Loaded {model_name}: {len(epochs)} epochs")
            else:
                print(f"✗ No data found for {model_name}")
        else:
            print(f"✗ File not found: {file_path}")
    
    # Create individual plots for each model
    for model_name, data in model_data.items():
        epochs = data['epochs']
        val_acc = data['val_acc']
        style = model_styles[model_name]
        
        # Create figure for this model
        plt.figure(figsize=(12, 8))
        
        # Plot the validation accuracy progression
        plt.plot(epochs, val_acc, 
                marker=style['marker'], 
                linewidth=4, 
                markersize=10,
                color=style['color'], 
                linestyle=style['linestyle'],
                alpha=0.8,
                markerfacecolor=style['color'],
                markeredgecolor='white',
                markeredgewidth=2)
        
        # Add value labels on points
        for epoch, acc in zip(epochs, val_acc):
            plt.annotate(f'{acc:.3f}', 
                        (epoch, acc), 
                        textcoords="offset points", 
                        xytext=(0,15), 
                        ha='center',
                        fontsize=11,
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=style['color']))
        
        # Calculate statistics
        start_acc = val_acc[0]
        end_acc = val_acc[-1]
        best_acc = max(val_acc)
        best_epoch = epochs[val_acc.index(best_acc)]
        total_growth = (end_acc - start_acc) * 100
        avg_improvement = total_growth / (len(val_acc) - 1) if len(val_acc) > 1 else 0
        
        # Customize the plot
        display_name = model_name.replace('_', ' ').replace('CNN ViT', 'CNN-ViT').replace('EfficientNet', '+EfficientNet')
        plt.title(f'{display_name} - Validation Accuracy Evolution\nEpochs 1-10', 
                  fontsize=18, fontweight='bold', pad=20, color=style['color'])
        plt.xlabel('Epoch Number', fontsize=14, fontweight='bold')
        plt.ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Set axis limits and ticks
        plt.xlim(0.5, 10.5)
        plt.ylim(0, 1)
        plt.xticks(range(1, 11), fontsize=12)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
        
        # Add horizontal reference lines
        for y in [0.5, 0.7, 0.8, 0.9]:
            plt.axhline(y=y, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add statistics text box
        stats_text = f"""Statistics:
Starting Accuracy: {start_acc:.4f}
Ending Accuracy: {end_acc:.4f}
Best Accuracy: {best_acc:.4f} (Epoch {best_epoch})
Total Growth: {total_growth:+.2f}%
Avg Improvement/Epoch: {avg_improvement:+.2f}%"""
        
        plt.text(0.02, 0.98, stats_text, 
                transform=plt.gca().transAxes, 
                fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8, edgecolor=style['color']))
        
        # Add trend line
        if len(epochs) > 1:
            z = np.polyfit(epochs, val_acc, 1)
            p = np.poly1d(z)
            plt.plot(epochs, p(epochs), 
                    color=style['color'], 
                    linestyle='--', 
                    alpha=0.5, 
                    linewidth=2,
                    label=f'Trend (slope: {z[0]:.4f})')
            plt.legend(loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        filename = f'{model_name}_evolution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"✓ Created {filename}")
        
        # Print model summary
        print(f"\n{display_name} Summary:")
        print(f"  Epochs: {len(epochs)}")
        print(f"  Starting Accuracy: {start_acc:.4f}")
        print(f"  Ending Accuracy: {end_acc:.4f}")
        print(f"  Best Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
        print(f"  Total Growth: {total_growth:+.2f}%")
        print(f"  Average Improvement/Epoch: {avg_improvement:+.2f}%")
        print("-" * 50)
    
    # Create a summary comparison plot
    print("\nCreating summary comparison plot...")
    plt.figure(figsize=(16, 10))
    
    for model_name, data in model_data.items():
        epochs = data['epochs']
        val_acc = data['val_acc']
        style = model_styles[model_name]
        display_name = model_name.replace('_', ' ').replace('CNN ViT', 'CNN-ViT').replace('EfficientNet', '+EfficientNet')
        
        plt.plot(epochs, val_acc, 
                marker=style['marker'], 
                linewidth=3, 
                markersize=8,
                color=style['color'], 
                linestyle=style['linestyle'],
                label=display_name,
                alpha=0.8)
    
    plt.title('All Models - Validation Accuracy Evolution Comparison\nEpochs 1-10', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch Number', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Set axis limits and ticks
    plt.xlim(0.5, 10.5)
    plt.ylim(0, 1)
    plt.xticks(range(1, 11), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    
    # Add horizontal reference lines
    for y in [0.5, 0.7, 0.8, 0.9]:
        plt.axhline(y=y, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ Created all_models_comparison.png")

if __name__ == "__main__":
    create_individual_model_plots()

