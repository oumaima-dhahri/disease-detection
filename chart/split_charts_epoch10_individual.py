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
            r'Epoch (\d+)/10 \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)',
            # ConvNeXt pattern
            r'Epoch (\d+)/10 \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)',
            # Hybrid CNN-ViT pattern
            r'Epoch (\d+)/10 \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)',
            # YOLOv9 + EfficientNet pattern
            r'Epoch (\d+)/10:\s+Train Loss: ([\d.]+), Train Acc: ([\d.]+)%\s+Val Loss: ([\d.]+), Val Acc: ([\d.]+)%',
            # ProtoPNet pattern
            r'Epoch (\d+)/10\s+Train Loss: ([\d.]+) Acc: ([\d.]+)\s+Val Loss: ([\d.]+) Acc: ([\d.]+)',
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

def create_epoch10_individual_chart_files():
    """Create separate image files for each chart type for epoch 10 data"""
    
    # Define model log files for epoch 10
    model_logs = {
        'SC-ConvNeXt': 'epoch10/output trainig/Train sc convnext.txt',
        'ConvNeXt': 'epoch10/output trainig/train convnext.txt',
        'Hybrid CNN-ViT': 'epoch10/output trainig/train hybrid cnn.txt',
        'Hybrid V2': 'epoch10/output trainig/train hybrid v2.txt',
        'YOLOv9 + EfficientNet': 'epoch10/output trainig/train yolo9 efficient net b3.txt',
        'ProtoPNet': 'epoch10/output trainig/train protopnet.txt'
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
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create individual chart files for each metric
    print("\nCreating individual chart files for Epoch 10...")
    
    # 1. Training Accuracy Chart
    plt.figure(figsize=(12, 8))
    for i, (model_name, data) in enumerate(model_data.items()):
        plt.plot(data['epochs'], data['train_accs'], 
                label=f"{model_name} (Final: {data['train_accs'][-1]:.3f})", 
                color=colors[i], linewidth=3, marker='o', markersize=6)
    
    plt.title('Training Accuracy Over Epochs - All Models (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Training Accuracy', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('epoch10_01_training_accuracy_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: epoch10_01_training_accuracy_chart.png")
    plt.close()
    
    # 2. Validation Accuracy Chart
    plt.figure(figsize=(12, 8))
    for i, (model_name, data) in enumerate(model_data.items()):
        plt.plot(data['epochs'], data['val_accs'], 
                label=f"{model_name} (Final: {data['val_accs'][-1]:.3f})", 
                color=colors[i], linewidth=3, marker='s', markersize=6)
    
    plt.title('Validation Accuracy Over Epochs - All Models (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('epoch10_02_validation_accuracy_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: epoch10_02_validation_accuracy_chart.png")
    plt.close()
    
    # 3. Training Loss Chart
    plt.figure(figsize=(12, 8))
    for i, (model_name, data) in enumerate(model_data.items()):
        plt.plot(data['epochs'], data['train_losses'], 
                label=f"{model_name} (Final: {data['train_losses'][-1]:.3f})", 
                color=colors[i], linewidth=3, marker='^', markersize=6)
    
    plt.title('Training Loss Over Epochs - All Models (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('epoch10_03_training_loss_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: epoch10_03_training_loss_chart.png")
    plt.close()
    
    # 4. Validation Loss Chart
    plt.figure(figsize=(12, 8))
    for i, (model_name, data) in enumerate(model_data.items()):
        plt.plot(data['epochs'], data['val_losses'], 
                label=f"{model_name} (Final: {data['val_losses'][-1]:.3f})", 
                color=colors[i], linewidth=3, marker='D', markersize=6)
    
    plt.title('Validation Loss Over Epochs - All Models (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Validation Loss', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('epoch10_04_validation_loss_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: epoch10_04_validation_loss_chart.png")
    plt.close()
    
    # 5. Final Performance Comparison Chart
    plt.figure(figsize=(14, 8))
    models = list(model_data.keys())
    final_train_accs = [model_data[model]['train_accs'][-1] for model in models]
    final_val_accs = [model_data[model]['val_accs'][-1] for model in models]
    test_accs = [model_data[model]['test_accs'][0] if model_data[model]['test_accs'] else 0 for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = plt.bar(x - width, final_train_accs, width, label='Final Train Accuracy', alpha=0.8, color='lightblue')
    bars2 = plt.bar(x, final_val_accs, width, label='Final Val Accuracy', alpha=0.8, color='lightgreen')
    bars3 = plt.bar(x + width, test_accs, width, label='Test Accuracy', alpha=0.8, color='lightcoral')
    
    plt.title('Final Performance Comparison - All Models (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(x, [model.replace(' ', '\n') for model in models], fontsize=10, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('epoch10_05_final_performance_comparison_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: epoch10_05_final_performance_comparison_chart.png")
    plt.close()
    
    # 6. Best Validation Accuracy Ranking Chart
    plt.figure(figsize=(12, 8))
    best_val_accs = [max(model_data[model]['val_accs']) for model in models]
    best_epochs = [model_data[model]['epochs'][model_data[model]['val_accs'].index(max(model_data[model]['val_accs']))] for model in models]
    
    # Sort by best validation accuracy
    sorted_indices = np.argsort(best_val_accs)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_accs = [best_val_accs[i] for i in sorted_indices]
    sorted_epochs = [best_epochs[i] for i in sorted_indices]
    
    bars = plt.barh(range(len(sorted_models)), sorted_accs, alpha=0.8, color=[colors[i] for i in sorted_indices])
    plt.title('Best Validation Accuracy Ranking - All Models (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Best Validation Accuracy', fontsize=14)
    plt.ylabel('Models', fontsize=14)
    plt.yticks(range(len(sorted_models)), [model.replace(' ', '\n') for model in sorted_models], fontsize=10)
    plt.grid(True, alpha=0.3, axis='x')
    plt.xlim(0, 1)
    
    # Add value labels on bars
    for i, (bar, epoch) in enumerate(zip(bars, sorted_epochs)):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}\n(Epoch {epoch})', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('epoch10_06_best_validation_accuracy_ranking_chart.png', dpi=300, bbox_inches='tight')
    print("Saved: epoch10_06_best_validation_accuracy_ranking_chart.png")
    plt.close()
    
    return model_data

def create_epoch10_individual_model_charts():
    """Create separate chart files for each model for epoch 10 data"""
    
    # Define model log files for epoch 10
    model_logs = {
        'SC-ConvNeXt': 'epoch10/output trainig/Train sc convnext.txt',
        'ConvNeXt': 'epoch10/output trainig/train convnext.txt',
        'Hybrid CNN-ViT': 'epoch10/output trainig/train hybrid cnn.txt',
        'Hybrid V2': 'epoch10/output trainig/train hybrid v2.txt',
        'YOLOv9 + EfficientNet': 'epoch10/output trainig/train yolo9 efficient net b3.txt',
        'ProtoPNet': 'epoch10/output trainig/train protopnet.txt'
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
    
    print("\nCreating individual model chart files for Epoch 10...")
    
    # Create separate charts for each model
    for model_name, data in model_data.items():
        safe_name = model_name.replace(' ', '_').replace('+', 'plus').replace('-', '_')
        
        # 1. Training Accuracy for this model
        plt.figure(figsize=(10, 6))
        plt.plot(data['epochs'], data['train_accs'], 'b-', linewidth=3, marker='o', markersize=6)
        plt.title(f'{model_name} - Training Accuracy (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Training Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'epoch10_{safe_name}_training_accuracy.png', dpi=300, bbox_inches='tight')
        print(f"Saved: epoch10_{safe_name}_training_accuracy.png")
        plt.close()
        
        # 2. Validation Accuracy for this model
        plt.figure(figsize=(10, 6))
        plt.plot(data['epochs'], data['val_accs'], 'r-', linewidth=3, marker='s', markersize=6)
        plt.title(f'{model_name} - Validation Accuracy (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Validation Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'epoch10_{safe_name}_validation_accuracy.png', dpi=300, bbox_inches='tight')
        print(f"Saved: epoch10_{safe_name}_validation_accuracy.png")
        plt.close()
        
        # 3. Training Loss for this model
        plt.figure(figsize=(10, 6))
        plt.plot(data['epochs'], data['train_losses'], 'g-', linewidth=3, marker='^', markersize=6)
        plt.title(f'{model_name} - Training Loss (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Training Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'epoch10_{safe_name}_training_loss.png', dpi=300, bbox_inches='tight')
        print(f"Saved: epoch10_{safe_name}_training_loss.png")
        plt.close()
        
        # 4. Validation Loss for this model
        plt.figure(figsize=(10, 6))
        plt.plot(data['epochs'], data['val_losses'], 'm-', linewidth=3, marker='D', markersize=6)
        plt.title(f'{model_name} - Validation Loss (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Validation Loss', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'epoch10_{safe_name}_validation_loss.png', dpi=300, bbox_inches='tight')
        print(f"Saved: epoch10_{safe_name}_validation_loss.png")
        plt.close()
        
        # 5. Combined Accuracy for this model
        plt.figure(figsize=(10, 6))
        plt.plot(data['epochs'], data['train_accs'], 'b-', label='Training Accuracy', linewidth=3, marker='o', markersize=6)
        plt.plot(data['epochs'], data['val_accs'], 'r-', label='Validation Accuracy', linewidth=3, marker='s', markersize=6)
        plt.title(f'{model_name} - Accuracy Comparison (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'epoch10_{safe_name}_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: epoch10_{safe_name}_accuracy_comparison.png")
        plt.close()
        
        # 6. Combined Loss for this model
        plt.figure(figsize=(10, 6))
        plt.plot(data['epochs'], data['train_losses'], 'g-', label='Training Loss', linewidth=3, marker='^', markersize=6)
        plt.plot(data['epochs'], data['val_losses'], 'm-', label='Validation Loss', linewidth=3, marker='D', markersize=6)
        plt.title(f'{model_name} - Loss Comparison (Epoch 10)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'epoch10_{safe_name}_loss_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: epoch10_{safe_name}_loss_comparison.png")
        plt.close()

if __name__ == "__main__":
    print("Creating individual chart files for Epoch 10...")
    
    print("\n1. Creating metric comparison charts for Epoch 10...")
    model_data = create_epoch10_individual_chart_files()
    
    print("\n2. Creating individual model charts for Epoch 10...")
    create_epoch10_individual_model_charts()
    
    print("\nAll individual chart files for Epoch 10 generated successfully!")
    print("\nGenerated files:")
    print("- 6 metric comparison charts (epoch10_01-06)")
    print("- 36 individual model charts (6 models × 6 chart types each)")
