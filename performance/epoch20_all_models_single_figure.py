import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
import os

def extract_epoch_data_from_log(log_file_path):
    """Extract epoch data from training log files"""
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_accs = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Different patterns for different model formats
        patterns = [
            # SC-ConvNeXt pattern
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
                    if 'yolo9' in str(log_file_path).lower():
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
        
        # Extract test accuracy from classification report
        test_acc_patterns = [
            r'accuracy\s+([\d.]+)',
            r'Test Accuracy: ([\d.]+)%',
            r'Test set predictions complete.*?accuracy\s+([\d.]+)',
            r'Final Test Accuracy: ([\d.]+)%'
        ]
        
        for pattern in test_acc_patterns:
            test_matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            if test_matches:
                test_acc = float(test_matches[-1])  # Get the last match
                if test_acc > 1:  # If it's a percentage
                    test_acc = test_acc / 100.0
                test_accs = [test_acc] * len(epochs)  # Same test acc for all epochs
                break
                
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        
    return epochs, train_losses, train_accs, val_losses, val_accs, test_accs

def create_epoch20_all_models_single_figure():
    """Create a single comprehensive figure with all models evaluation"""
    
    # Define model log files
    model_logs = {
        'SC-ConvNeXt': 'epoch20/output trainig/Train sc convnext.txt',
        'ConvNeXt': 'epoch20/output trainig/train convnext.txt',
        'Hybrid CNN-ViT': 'epoch20/output trainig/train hybrid cnn.txt',
        'Hybrid V2': 'epoch20/output trainig/train hybrid v2.txt',
        'YOLOv9 + EfficientNet': 'epoch20/output trainig/train yolo9 efficient net b3.txt',
        'ProtoPNet': 'epoch20/output trainig/train protopnet.txt'
    }
    
    # Extract data for all models
    model_data = {}
    for model_name, log_path in model_logs.items():
        print(f"Processing {model_name}...")
        epochs, train_losses, train_accs, val_losses, val_accs, test_accs = extract_epoch_data_from_log(log_path)
        if epochs:
            model_data[model_name] = {
                'epochs': epochs,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'test_accs': test_accs
            }
            print(f"  Found {len(epochs)} epochs, Test Acc: {test_accs[0] if test_accs else 'N/A'}")
        else:
            print(f"  No epoch data found")
    
    # Create a single comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # Define colors and styles for all models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # Plot 1: Training Accuracy (Top Left)
    ax1 = plt.subplot(2, 3, 1)
    for i, (model_name, data) in enumerate(model_data.items()):
        ax1.plot(data['epochs'], data['train_accs'], 
                label=f"{model_name}\n(Final: {data['train_accs'][-1]:.3f})", 
                color=colors[i], linewidth=2.5, 
                linestyle=linestyles[i], marker=markers[i], markersize=4, markevery=2)
    
    ax1.set_title('Training Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Accuracy', fontsize=12)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Validation Accuracy (Top Middle)
    ax2 = plt.subplot(2, 3, 2)
    for i, (model_name, data) in enumerate(model_data.items()):
        ax2.plot(data['epochs'], data['val_accs'], 
                label=f"{model_name}\n(Final: {data['val_accs'][-1]:.3f})", 
                color=colors[i], linewidth=2.5, 
                linestyle=linestyles[i], marker=markers[i], markersize=4, markevery=2)
    
    ax2.set_title('Validation Accuracy Over Epochs', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Training Loss (Top Right)
    ax3 = plt.subplot(2, 3, 3)
    for i, (model_name, data) in enumerate(model_data.items()):
        ax3.plot(data['epochs'], data['train_losses'], 
                label=f"{model_name}\n(Final: {data['train_losses'][-1]:.3f})", 
                color=colors[i], linewidth=2.5, 
                linestyle=linestyles[i], marker=markers[i], markersize=4, markevery=2)
    
    ax3.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Loss', fontsize=12)
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Validation Loss (Bottom Left)
    ax4 = plt.subplot(2, 3, 4)
    for i, (model_name, data) in enumerate(model_data.items()):
        ax4.plot(data['epochs'], data['val_losses'], 
                label=f"{model_name}\n(Final: {data['val_losses'][-1]:.3f})", 
                color=colors[i], linewidth=2.5, 
                linestyle=linestyles[i], marker=markers[i], markersize=4, markevery=2)
    
    ax4.set_title('Validation Loss Over Epochs', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Loss', fontsize=12)
    ax4.legend(fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: Final Performance Comparison (Bottom Middle)
    ax5 = plt.subplot(2, 3, 5)
    models = list(model_data.keys())
    final_train_accs = [model_data[model]['train_accs'][-1] for model in models]
    final_val_accs = [model_data[model]['val_accs'][-1] for model in models]
    test_accs = [model_data[model]['test_accs'][0] if model_data[model]['test_accs'] else 0 for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax5.bar(x - width, final_train_accs, width, label='Final Train', alpha=0.8, color='lightblue')
    bars2 = ax5.bar(x, final_val_accs, width, label='Final Val', alpha=0.8, color='lightgreen')
    bars3 = ax5.bar(x + width, test_accs, width, label='Test', alpha=0.8, color='lightcoral')
    
    ax5.set_title('Final Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax5.set_xlabel('Models', fontsize=12)
    ax5.set_ylabel('Accuracy', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels([model.replace(' ', '\n') for model in models], fontsize=8, rotation=45)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars3:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Best Validation Accuracy Ranking (Bottom Right)
    ax6 = plt.subplot(2, 3, 6)
    best_val_accs = [max(model_data[model]['val_accs']) for model in models]
    best_epochs = [model_data[model]['epochs'][model_data[model]['val_accs'].index(max(model_data[model]['val_accs']))] for model in models]
    
    # Sort by best validation accuracy
    sorted_indices = np.argsort(best_val_accs)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_accs = [best_val_accs[i] for i in sorted_indices]
    sorted_epochs = [best_epochs[i] for i in sorted_indices]
    
    bars = ax6.barh(range(len(sorted_models)), sorted_accs, alpha=0.8, color=[colors[i] for i in sorted_indices])
    ax6.set_title('Best Validation Accuracy Ranking', fontsize=14, fontweight='bold', pad=15)
    ax6.set_xlabel('Best Validation Accuracy', fontsize=12)
    ax6.set_ylabel('Models', fontsize=12)
    ax6.set_yticks(range(len(sorted_models)))
    ax6.set_yticklabels([model.replace(' ', '\n') for model in sorted_models], fontsize=8)
    ax6.grid(True, alpha=0.3, axis='x')
    ax6.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, (bar, epoch) in enumerate(zip(bars, sorted_epochs)):
        width = bar.get_width()
        ax6.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}\n(Epoch {epoch})', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('epoch20_all_models_single_figure.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary table
    print("\n" + "="*120)
    print("EPOCH 20 ALL MODELS EVALUATION SUMMARY")
    print("="*120)
    
    summary_data = []
    for model_name, data in model_data.items():
        best_val_acc = max(data['val_accs'])
        best_epoch = data['epochs'][data['val_accs'].index(best_val_acc)]
        test_acc = data['test_accs'][0] if data['test_accs'] else 0
        overfitting_gap = data['train_accs'][-1] - data['val_accs'][-1]
        
        summary_data.append({
            'Model': model_name,
            'Total Epochs': len(data['epochs']),
            'Final Train Acc': f"{data['train_accs'][-1]:.4f}",
            'Final Val Acc': f"{data['val_accs'][-1]:.4f}",
            'Best Val Acc': f"{best_val_acc:.4f}",
            'Best Epoch': best_epoch,
            'Test Acc': f"{test_acc:.4f}" if test_acc > 0 else "N/A",
            'Overfitting Gap': f"{overfitting_gap:.4f}",
            'Final Train Loss': f"{data['train_losses'][-1]:.4f}",
            'Final Val Loss': f"{data['val_losses'][-1]:.4f}"
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Performance ranking
    print("\n" + "="*80)
    print("PERFORMANCE RANKING (by Best Validation Accuracy)")
    print("="*80)
    
    ranking_data = sorted(summary_data, key=lambda x: float(x['Best Val Acc']), reverse=True)
    for i, model in enumerate(ranking_data, 1):
        print(f"{i}. {model['Model']}: {model['Best Val Acc']} (Epoch {model['Best Epoch']}) - Test: {model['Test Acc']}")
    
    return model_data

if __name__ == "__main__":
    print("Creating single figure with all models evaluation...")
    model_data = create_epoch20_all_models_single_figure()
    print("\nSingle figure with all models evaluation generated successfully!")
    print("Saved as: epoch20_all_models_single_figure.png")

