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

def create_individual_model_plots():
    """Create separate plots for each model"""
    
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
    
    # Create individual plots for each model
    for model_name, data in model_data.items():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} - Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Training and Validation Accuracy
        ax1.plot(data['epochs'], data['train_accs'], 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        ax1.plot(data['epochs'], data['val_accs'], 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        ax1.set_title('Accuracy Over Epochs', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Training and Validation Loss
        ax2.plot(data['epochs'], data['train_losses'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
        ax2.plot(data['epochs'], data['val_losses'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Loss Over Epochs', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Training Accuracy Only
        ax3.plot(data['epochs'], data['train_accs'], 'g-', label='Training Accuracy', linewidth=3, marker='o', markersize=5)
        ax3.set_title('Training Accuracy', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Validation Accuracy Only
        ax4.plot(data['epochs'], data['val_accs'], 'm-', label='Validation Accuracy', linewidth=3, marker='s', markersize=5)
        ax4.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save individual plot
        safe_name = model_name.replace(' ', '_').replace('+', 'plus')
        filename = f'individual_{safe_name}_curves.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()
    
    return model_data

def create_comparison_plots():
    """Create side-by-side comparison plots for all models"""
    
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
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('All Models Comparison - Separated Curves', fontsize=18, fontweight='bold')
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Training Accuracy Comparison
    for i, (model_name, data) in enumerate(model_data.items()):
        ax1.plot(data['epochs'], data['train_accs'], 
                label=model_name, color=colors[i], linewidth=2.5, marker='o', markersize=4)
    
    ax1.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Validation Accuracy Comparison
    for i, (model_name, data) in enumerate(model_data.items()):
        ax2.plot(data['epochs'], data['val_accs'], 
                label=model_name, color=colors[i], linewidth=2.5, marker='s', markersize=4)
    
    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Training Loss Comparison
    for i, (model_name, data) in enumerate(model_data.items()):
        ax3.plot(data['epochs'], data['train_losses'], 
                label=model_name, color=colors[i], linewidth=2.5, marker='^', markersize=4)
    
    ax3.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Loss')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Validation Loss Comparison
    for i, (model_name, data) in enumerate(model_data.items()):
        ax4.plot(data['epochs'], data['val_losses'], 
                label=model_name, color=colors[i], linewidth=2.5, marker='D', markersize=4)
    
    ax4.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('all_models_comparison_separated.png', dpi=300, bbox_inches='tight')
    print("Saved: all_models_comparison_separated.png")
    plt.show()

def create_single_metric_plots():
    """Create separate plots for each metric across all models"""
    
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
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create separate plots for each metric
    metrics = [
        ('train_accs', 'Training Accuracy', 'training_accuracy_comparison.png'),
        ('val_accs', 'Validation Accuracy', 'validation_accuracy_comparison.png'),
        ('train_losses', 'Training Loss', 'training_loss_comparison.png'),
        ('val_losses', 'Validation Loss', 'validation_loss_comparison.png')
    ]
    
    for metric_key, metric_title, filename in metrics:
        plt.figure(figsize=(12, 8))
        
        for i, (model_name, data) in enumerate(model_data.items()):
            plt.plot(data['epochs'], data[metric_key], 
                    label=model_name, color=colors[i], linewidth=3, marker='o', markersize=5)
        
        plt.title(f'{metric_title} - All Models', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric_title, fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        if 'loss' in metric_key.lower():
            plt.yscale('log')
        else:
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()

if __name__ == "__main__":
    print("Creating separated curve plots...")
    
    print("\n1. Creating individual model plots...")
    model_data = create_individual_model_plots()
    
    print("\n2. Creating comparison plots...")
    create_comparison_plots()
    
    print("\n3. Creating single metric plots...")
    create_single_metric_plots()
    
    print("\nAll separated curve plots generated successfully!")
