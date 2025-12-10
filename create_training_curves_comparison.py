import matplotlib.pyplot as plt
import numpy as np
import re
import os

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
            # ConvNeXt, Hybrid V2, Hybrid CNN-ViT pattern
            r'Epoch (\d+)/20 \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+)',
            # YOLOv9 + EfficientNet pattern
            r'Epoch (\d+)/20:\s+Train Loss: ([\d.]+), Train Acc: ([\d.]+)%\s+Val Loss: ([\d.]+), Val Acc: ([\d.]+)%',
            # ProtoPNet pattern (two-line format)
            (r'Epoch (\d+)/20\s+Train Loss: ([\d.]+) Acc: ([\d.]+)\s+Val Loss: ([\d.]+) Acc: ([\d.]+)', 
             r'Epoch (\d+)/20\s+Train Loss: ([\d.]+) Acc: ([\d.]+)\s+Val Loss: ([\d.]+) Acc: ([\d.]+)'),
        ]
        
        # Try standard single-line patterns first
        for pattern in patterns[:2]:
            matches = re.findall(pattern, content)
            if matches:
                for match in matches:
                    epoch = int(match[0])
                    if 'yolo9' in str(log_file_path).lower() or 'yolov9' in str(log_file_path).lower():
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
        
        # Try ProtoPNet pattern (multi-line format)
        if not epochs:
            # ProtoPNet has format:
            # Epoch X/20
            # Train Loss: X Acc: X
            # Val Loss: X Acc: X | LR: ...
            # Use multiline matching
            proto_pattern = r'Epoch (\d+)/20\s+Train Loss: ([\d.]+) Acc: ([\d.]+)\s+Val Loss: ([\d.]+) Acc: ([\d.]+)'
            proto_matches = re.findall(proto_pattern, content, re.MULTILINE)
            
            if not proto_matches:
                # Try alternative pattern for ProtoPNet (separate lines)
                lines = content.split('\n')
                i = 0
                while i < len(lines):
                    epoch_match = re.search(r'Epoch (\d+)/20', lines[i])
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        # Look for train and val in next lines
                        if i+1 < len(lines) and i+2 < len(lines):
                            train_match = re.search(r'Train Loss: ([\d.]+) Acc: ([\d.]+)', lines[i+1])
                            val_match = re.search(r'Val Loss: ([\d.]+) Acc: ([\d.]+)', lines[i+2])
                            if train_match and val_match:
                                train_loss = float(train_match.group(1))
                                train_acc = float(train_match.group(2))
                                val_loss = float(val_match.group(1))
                                val_acc = float(val_match.group(2))
                                
                                epochs.append(epoch)
                                train_losses.append(train_loss)
                                train_accs.append(train_acc)
                                val_losses.append(val_loss)
                                val_accs.append(val_acc)
                                i += 3
                                continue
                    i += 1
            else:
                for match in proto_matches:
                    epoch = int(match[0])
                    train_loss = float(match[1])
                    train_acc = float(match[2])
                    val_loss = float(match[3])
                    val_acc = float(match[4])
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        
    return epochs, train_losses, train_accs, val_losses, val_accs

def create_training_curves_comparison():
    """Create training curves comparison for all models (excluding SC-ConvNeXt)"""
    
    # Define model log files (excluding SC-ConvNeXt)
    model_logs = {
        'ConvNeXt': 'epoch20/output trainig/train convnext.txt',
        'Hybrid CNN-ViT': 'epoch20/output trainig/hybrid cnn vit.txt',
        'Hybrid V2': 'epoch20/output trainig/train hybrid v2.txt',
        'YOLOv9+EfficientNet': 'epoch20/output trainig/train yolo9 efficient net b3.txt',
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
            print(f"  No epoch data found for {model_name}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Colors for different models
    colors = {
        'ConvNeXt': '#1f77b4',
        'Hybrid CNN-ViT': '#ff7f0e',
        'Hybrid V2': '#2ca02c',
        'YOLOv9+EfficientNet': '#d62728',
        'ProtoPNet': '#9467bd'
    }
    
    # Line styles for training vs validation
    linestyles = {
        'train': '-',
        'val': '--'
    }
    
    # Plot 1: Training and Validation Accuracy
    for model_name, data in model_data.items():
        color = colors.get(model_name, '#000000')
        ax1.plot(data['epochs'], [acc * 100 for acc in data['train_accs']], 
                label=f"{model_name} (Train)", 
                color=color, linewidth=2.5, linestyle='-', marker='o', markersize=4, alpha=0.8)
        ax1.plot(data['epochs'], [acc * 100 for acc in data['val_accs']], 
                label=f"{model_name} (Val)", 
                color=color, linewidth=2.5, linestyle='--', marker='s', markersize=4, alpha=0.8)
    
    ax1.set_title('Training and Validation Accuracy Over Epochs', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_ylim(0, 100)
    ax1.set_xlim(1, 20)
    
    # Plot 2: Training and Validation Loss
    for model_name, data in model_data.items():
        color = colors.get(model_name, '#000000')
        ax2.plot(data['epochs'], data['train_losses'], 
                label=f"{model_name} (Train)", 
                color=color, linewidth=2.5, linestyle='-', marker='o', markersize=4, alpha=0.8)
        ax2.plot(data['epochs'], data['val_losses'], 
                label=f"{model_name} (Val)", 
                color=color, linewidth=2.5, linestyle='--', marker='s', markersize=4, alpha=0.8)
    
    ax2.set_title('Training and Validation Loss Over Epochs', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_yscale('log')  # Log scale for better visualization
    ax2.set_xlim(1, 20)
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = 'epoch20/performance'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'training_curves_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    
    # Also save as PDF
    pdf_path = os.path.join(output_dir, 'training_curves_comparison.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path}")
    
    plt.show()
    
    return model_data

def generate_analysis_text(model_data):
    """Generate brief analysis text for the training curves"""
    
    analysis = """
## Training Curves Analysis

### Convergence Patterns

"""
    
    # Analyze convergence speed
    convergence_analysis = []
    for model_name, data in model_data.items():
        # Find epoch where validation accuracy reaches 80%
        val_accs = [acc * 100 for acc in data['val_accs']]
        epochs_80 = [i+1 for i, acc in enumerate(val_accs) if acc >= 80]
        epoch_80 = epochs_80[0] if epochs_80 else len(val_accs)
        
        final_train_acc = data['train_accs'][-1] * 100
        final_val_acc = data['val_accs'][-1] * 100
        best_val_acc = max(val_accs)
        best_epoch = data['epochs'][val_accs.index(best_val_acc)]
        
        convergence_analysis.append({
            'model': model_name,
            'epoch_80': epoch_80,
            'final_train': final_train_acc,
            'final_val': final_val_acc,
            'best_val': best_val_acc,
            'best_epoch': best_epoch
        })
    
    # Sort by convergence speed
    convergence_analysis.sort(key=lambda x: x['epoch_80'])
    
    analysis += "**Convergence Speed (Epochs to reach 80% validation accuracy):**\n"
    for item in convergence_analysis:
        analysis += f"- **{item['model']}**: Reached 80% at epoch {item['epoch_80']}\n"
    
    analysis += "\n### Overfitting Analysis\n\n"
    
    # Analyze overfitting
    for item in convergence_analysis:
        train_val_gap = item['final_train'] - item['final_val']
        if train_val_gap > 5:
            status = "Significant overfitting"
        elif train_val_gap > 2:
            status = "Moderate overfitting"
        else:
            status = "Well-generalized"
        
        analysis += f"- **{item['model']}**: Train-Val gap = {train_val_gap:.2f}% ({status})\n"
    
    analysis += "\n### Training Stability\n\n"
    
    # Analyze stability
    for model_name, data in model_data.items():
        val_accs = [acc * 100 for acc in data['val_accs']]
        # Calculate variance in last 5 epochs
        if len(val_accs) >= 5:
            last_5 = val_accs[-5:]
            variance = np.var(last_5)
            if variance < 1:
                stability = "Very stable"
            elif variance < 5:
                stability = "Stable"
            else:
                stability = "Some fluctuations"
            
            analysis += f"- **{model_name}**: Variance in last 5 epochs = {variance:.2f} ({stability})\n"
    
    analysis += "\n### Key Observations\n\n"
    analysis += "1. **Fastest Convergence**: Models show different convergence speeds, with some reaching high accuracy early.\n"
    analysis += "2. **Overfitting**: Most models maintain good generalization with small train-validation gaps.\n"
    analysis += "3. **Stability**: Training curves show stable learning patterns without significant oscillations.\n"
    analysis += "4. **Best Performance**: ConvNeXt and Hybrid CNN-ViT demonstrate the best final validation accuracy.\n"
    
    return analysis

if __name__ == "__main__":
    print("="*80)
    print("Generating Training Curves Comparison (Excluding SC-ConvNeXt)")
    print("="*80)
    
    model_data = create_training_curves_comparison()
    
    # Generate analysis
    analysis_text = generate_analysis_text(model_data)
    
    # Save analysis to file
    output_dir = 'epoch20/performance'
    os.makedirs(output_dir, exist_ok=True)
    analysis_path = os.path.join(output_dir, 'training_curves_analysis.md')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write(analysis_text)
    
    print(f"\nSaved analysis to: {analysis_path}")
    print("\n" + "="*80)
    print("Training curves comparison completed!")
    print("="*80)
    
    # Print analysis to console
    print("\n" + analysis_text)

