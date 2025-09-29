#!/usr/bin/env python3
"""
Epoch 20 Curves - Create epoch-by-epoch curves for each model
Based on the training output files in epoch20/output trainig/
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pathlib import Path

def extract_standard_metrics(file_path):
    """Extract metrics from standard format logs (ConvNeXt, Hybrid_CNN_ViT, SC-ConvNeXt, Hybrid_V2)"""
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for standard format: Epoch X/10 | Train Loss: X.XXXX Acc: X.XXXX | Val Loss: X.XXXX Acc: X.XXXX
    pattern = r'Epoch (\d+)/10.*?Train Loss: ([\d.]+) Acc: ([\d.]+).*?Val Loss: ([\d.]+) Acc: ([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        epoch = int(match[0])
        epochs.append(epoch)
        train_loss.append(float(match[1]))
        train_acc.append(float(match[2]))
        val_loss.append(float(match[3]))
        val_acc.append(float(match[4]))
    
    return epochs, train_loss, train_acc, val_loss, val_acc

def extract_protopnet_metrics(file_path):
    """Extract metrics from ProtoPNet format logs"""
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Epoch') and '/10' in line:
            # Extract epoch number
            epoch_match = re.search(r'Epoch (\d+)/10', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)
                
                # Next line should have train metrics
                if i + 1 < len(lines):
                    train_line = lines[i + 1].strip()
                    train_match = re.search(r'Train Loss: ([\d.]+) Acc: ([\d.]+)', train_line)
                    if train_match:
                        train_loss.append(float(train_match.group(1)))
                        train_acc.append(float(train_match.group(2)))
                
                # Look for validation metrics
                if i + 2 < len(lines):
                    val_line = lines[i + 2].strip()
                    val_match = re.search(r'Val Loss: ([\d.]+) Acc: ([\d.]+)', val_line)
                    if val_match:
                        val_loss.append(float(val_match.group(1)))
                        val_acc.append(float(val_match.group(2)))
        i += 1
    
    return epochs, train_loss, train_acc, val_loss, val_acc

def extract_yolov9_metrics(file_path):
    """Extract metrics from YOLOv9 format logs"""
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for YOLOv9 format: Epoch X/10: Train Loss: X.XXXX, Train Acc: XX.XX% Val Loss: X.XXXX, Val Acc: XX.XX%
    pattern = r'Epoch (\d+)/10:.*?Train Loss: ([\d.]+), Train Acc: ([\d.]+)%.*?Val Loss: ([\d.]+), Val Acc: ([\d.]+)%'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for match in matches:
        epoch = int(match[0])
        epochs.append(epoch)
        train_loss.append(float(match[1]))
        train_acc.append(float(match[2]) / 100.0)  # Convert percentage to decimal
        val_loss.append(float(match[3]))
        val_acc.append(float(match[4]) / 100.0)  # Convert percentage to decimal
    
    return epochs, train_loss, train_acc, val_loss, val_acc

def create_epoch_curves(model_name, epochs, train_loss, train_acc, val_loss, val_acc, model_index):
    """Create epoch curves for each model"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name.replace("_", " ").replace("hybride", "Hybrid").replace("protopnet", "ProtoPNet").replace("sc-convnext", "SC-ConvNeXt").replace("yolov9 efficient", "YOLOv9+EffNet").title()} - Epoch Curves', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    color = colors[model_index % len(colors)]
    
    # Training Accuracy Curve
    axes[0, 0].plot(epochs, train_acc, 'o-', color=color, linewidth=3, markersize=8, 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=color, label='Training Accuracy')
    axes[0, 0].set_title('Training Accuracy Curve', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].set_ylim(0, 1)
    
    # Validation Accuracy Curve
    axes[0, 1].plot(epochs, val_acc, 's-', color=color, linewidth=3, markersize=8, 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=color, label='Validation Accuracy')
    axes[0, 1].set_title('Validation Accuracy Curve', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].set_ylim(0, 1)
    
    # Training Loss Curve
    axes[1, 0].plot(epochs, train_loss, '^-', color=color, linewidth=3, markersize=8, 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=color, label='Training Loss')
    axes[1, 0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=11)
    
    # Validation Loss Curve
    axes[1, 1].plot(epochs, val_loss, 'v-', color=color, linewidth=3, markersize=8, 
                    markerfacecolor='white', markeredgewidth=2, markeredgecolor=color, label='Validation Loss')
    axes[1, 1].set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=11)
    
    # Add performance statistics
    stats_text = f"""Performance Summary:
Epochs: {len(epochs)}
Final Train Acc: {train_acc[-1]:.4f}
Final Val Acc: {val_acc[-1]:.4f}
Best Val Acc: {max(val_acc):.4f}
Final Train Loss: {train_loss[-1]:.4f}
Final Val Loss: {val_loss[-1]:.4f}
Improvement: {val_acc[-1] - val_acc[0]:.4f}"""
    
    axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=10)
    
    plt.tight_layout()
    filename = f'{model_name}_epoch_curves.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved epoch curves for {model_name}: {filename}")
    
    return filename

def create_all_curves_comparison(all_models_data):
    """Create comparison of all models' epoch curves"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('All Models Epoch Curves Comparison', fontsize=18, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (model_name, data) in enumerate(all_models_data.items()):
        epochs, train_loss, train_acc, val_loss, val_acc = data
        color = colors[i % len(colors)]
        display_name = model_name.replace("_", " ").replace("hybride", "Hybrid").replace("protopnet", "ProtoPNet").replace("sc-convnext", "SC-ConvNeXt").replace("yolov9 efficient", "YOLOv9+EffNet").title()
        
        # Training Accuracy Curves
        axes[0, 0].plot(epochs, train_acc, 'o-', color=color, linewidth=2.5, markersize=6, 
                        markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color, label=display_name)
        
        # Validation Accuracy Curves
        axes[0, 1].plot(epochs, val_acc, 's-', color=color, linewidth=2.5, markersize=6, 
                        markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color, label=display_name)
        
        # Training Loss Curves
        axes[1, 0].plot(epochs, train_loss, '^-', color=color, linewidth=2.5, markersize=6, 
                        markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color, label=display_name)
        
        # Validation Loss Curves
        axes[1, 1].plot(epochs, val_loss, 'v-', color=color, linewidth=2.5, markersize=6, 
                        markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=color, label=display_name)
    
    # Configure subplots
    axes[0, 0].set_title('Training Accuracy Curves', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[0, 0].set_ylim(0, 1)
    
    axes[0, 1].set_title('Validation Accuracy Curves', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[0, 1].set_ylim(0, 1)
    
    axes[1, 0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    axes[1, 1].set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    filename = 'all_models_epoch_curves_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved all models comparison: {filename}")
    
    return filename

def main():
    """Main function to create epoch curves from epoch 20 training outputs"""
    print("📈 Creating Epoch Curves from Epoch 20 Training Outputs...")
    print("=" * 70)
    
    # Define model files and their extraction functions
    models = {
        'ConvNeXt': ('epoch20/output trainig/convnext.txt', extract_standard_metrics),
        'Hybrid_CNN_ViT': ('epoch20/output trainig/hybride_cnn_vit.txt', extract_standard_metrics),
        'Hybrid_V2': ('epoch20/output trainig/hybride_v2.txt', extract_standard_metrics),
        'ProtoPNet': ('epoch20/output trainig/protopnet .txt', extract_protopnet_metrics),
        'SC_ConvNeXt': ('epoch20/output trainig/sc-convnext.txt', extract_standard_metrics),
        'YOLOv9_EfficientNet': ('epoch20/output trainig/yolov9 and efficient b3.txt', extract_yolov9_metrics)
    }
    
    all_models_data = {}
    individual_curves = []
    
    # Process each model
    for i, (model_name, (file_path, extract_func)) in enumerate(models.items()):
        print(f"\n📊 Processing {model_name}...")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        try:
            epochs, train_loss, train_acc, val_loss, val_acc = extract_func(file_path)
            
            if not epochs:
                print(f"❌ No epoch data found in {file_path}")
                continue
            
            print(f"✅ Extracted {len(epochs)} epochs of data")
            print(f"   Epochs: {epochs}")
            print(f"   Final Val Acc: {val_acc[-1]:.4f}")
            print(f"   Best Val Acc: {max(val_acc):.4f}")
            print(f"   Improvement: {val_acc[-1] - val_acc[0]:.4f}")
            
            # Store data for comparison plot
            all_models_data[model_name] = (epochs, train_loss, train_acc, val_loss, val_acc)
            
            # Create individual epoch curves
            curve_file = create_epoch_curves(model_name, epochs, train_loss, train_acc, val_loss, val_acc, i)
            individual_curves.append(curve_file)
            
        except Exception as e:
            print(f"❌ Error processing {model_name}: {str(e)}")
            continue
    
    # Create comparison plot
    if all_models_data:
        print(f"\n📈 Creating comparison curves for {len(all_models_data)} models...")
        comparison_curves = create_all_curves_comparison(all_models_data)
        
        # Summary
        print("\n" + "=" * 70)
        print("📋 SUMMARY - Epoch Curves Analysis")
        print("=" * 70)
        print(f"✅ Processed {len(all_models_data)} models successfully")
        print(f"✅ Generated {len(individual_curves)} individual curve plots")
        print(f"✅ Generated 1 comparison curve plot")
        print("\n📁 Generated Files:")
        for curve in individual_curves:
            print(f"   • {curve}")
        print(f"   • {comparison_curves}")
        
        print("\n📊 Performance Ranking (Final Validation Accuracy):")
        sorted_models = sorted(all_models_data.items(), key=lambda x: x[1][3][-1], reverse=True)
        for i, (model_name, data) in enumerate(sorted_models, 1):
            epochs, train_loss, train_acc, val_loss, val_acc = data
            print(f"   {i}. {model_name}: {val_acc[-1]:.4f} (Best: {max(val_acc):.4f})")
        
        print("\n⚠️  NOTE:")
        print("   The 'epoch20' directory contains only 10 epochs of data, not 20.")
        print("   The curves show the actual available training data.")
        
    else:
        print("❌ No valid model data found!")

if __name__ == "__main__":
    main()

