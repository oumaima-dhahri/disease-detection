"""
Training Progress Analysis - Loss and Accuracy Curves for ConvNeXt

This script generates training progress visualizations (loss and accuracy curves)
from saved training logs in the saved_models_and_data directory.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# -----------------------------
# Configuration
# -----------------------------
SAVE_DIR = '../../saved_models_and_data'
OUTPUT_DIR = '../../saved_models_and_data/convnext_training_curves'
MODEL_TYPES = ['grid_search_convnext', 'grid_search_msconvnext']  # Add more if needed

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Training Logs
# -----------------------------
def load_training_logs(base_dir: str, model_types: List[str]) -> Dict:
    """Load all training logs from grid search directories"""
    all_logs = {}
    base_path = Path(base_dir)
    
    for model_type in model_types:
        grid_search_dir = base_path / model_type
        if not grid_search_dir.exists():
            print(f"⚠️  Directory not found: {grid_search_dir}")
            continue
        
        print(f"📂 Searching in: {grid_search_dir}")
        
        # Find all trial directories
        for trial_dir in grid_search_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            
            train_log_path = trial_dir / "train_log.json"
            if train_log_path.exists():
                try:
                    with open(train_log_path, 'r') as f:
                        log_data = json.load(f)
                    
                    trial_name = trial_dir.name
                    full_name = f"{model_type}/{trial_name}"
                    all_logs[full_name] = {
                        'data': log_data,
                        'model_type': model_type,
                        'trial_name': trial_name
                    }
                    print(f"  ✓ Loaded: {full_name} ({len(log_data)} epochs)")
                except Exception as e:
                    print(f"  ✗ Error loading {trial_dir}: {e}")
            else:
                print(f"  ⚠️  No train_log.json in {trial_dir}")
    
    return all_logs

# -----------------------------
# Extract Metrics from Logs
# -----------------------------
def extract_metrics(log_data: List[Dict]) -> Tuple[List, List, List, List]:
    """Extract epochs, train_loss, train_acc, val_loss, val_acc from log data"""
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for entry in log_data:
        epochs.append(entry.get('epoch', len(epochs) + 1))
        train_losses.append(entry.get('train_loss', 0))
        train_accs.append(entry.get('train_acc', 0))
        val_losses.append(entry.get('val_loss', 0))
        val_accs.append(entry.get('val_acc', 0))
    
    return epochs, train_losses, train_accs, val_losses, val_accs

# -----------------------------
# Create Individual Trial Curves
# -----------------------------
def create_individual_curves(all_logs: Dict, output_dir: str):
    """Create separate curves for each trial"""
    print("\n" + "="*80)
    print("Creating individual trial curves...")
    print("="*80)
    
    for full_name, log_info in all_logs.items():
        log_data = log_info['data']
        trial_name = log_info['trial_name']
        model_type = log_info['model_type']
        
        epochs, train_losses, train_accs, val_losses, val_accs = extract_metrics(log_data)
        
        if not epochs:
            continue
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Progress: {trial_name}\n({model_type})', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Plot 1: Training & Validation Loss
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # Plot 2: Training & Validation Accuracy
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(left=0)
        
        # Plot 3: Loss Comparison (Stacked)
        ax3.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.7)
        ax3.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.7)
        ax3.fill_between(epochs, train_losses, alpha=0.3, color='blue')
        ax3.fill_between(epochs, val_losses, alpha=0.3, color='red')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Loss Comparison (Filled)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(left=0)
        
        # Plot 4: Accuracy Comparison (Stacked)
        ax4.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, alpha=0.7)
        ax4.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, alpha=0.7)
        ax4.fill_between(epochs, train_accs, alpha=0.3, color='blue')
        ax4.fill_between(epochs, val_accs, alpha=0.3, color='red')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.set_title('Accuracy Comparison (Filled)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        ax4.set_xlim(left=0)
        
        # Add statistics text
        stats_text = (
            f"Final Training Loss: {train_losses[-1]:.4f}\n"
            f"Final Validation Loss: {val_losses[-1]:.4f}\n"
            f"Final Training Accuracy: {train_accs[-1]:.4f}\n"
            f"Final Validation Accuracy: {val_accs[-1]:.4f}\n"
            f"Best Validation Accuracy: {max(val_accs):.4f} (Epoch {epochs[val_accs.index(max(val_accs))]})"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=9, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        safe_trial_name = trial_name.replace(' ', '_').replace('/', '_')
        output_path = Path(output_dir) / f"{safe_trial_name}_training_curves.png"
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

# -----------------------------
# Create Comparative Curves
# -----------------------------
def create_comparative_curves(all_logs: Dict, output_dir: str):
    """Create comparative curves for all trials"""
    print("\n" + "="*80)
    print("Creating comparative curves...")
    print("="*80)
    
    if not all_logs:
        print("⚠️  No logs to compare")
        return
    
    # Separate by model type
    convnext_logs = {k: v for k, v in all_logs.items() if 'grid_search_convnext' in k}
    msconvnext_logs = {k: v for k, v in all_logs.items() if 'grid_search_msconvnext' in k}
    
    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Colors for different trials
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot 1: All Trials - Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (full_name, log_info) in enumerate(all_logs.items()):
        epochs, train_losses, _, _, _ = extract_metrics(log_info['data'])
        label = log_info['trial_name']
        ax1.plot(epochs, train_losses, label=label, linewidth=2, 
                color=colors[i % len(colors)], marker='o', markersize=3, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss - All Trials', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Plot 2: All Trials - Validation Loss
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (full_name, log_info) in enumerate(all_logs.items()):
        epochs, _, _, val_losses, _ = extract_metrics(log_info['data'])
        label = log_info['trial_name']
        ax2.plot(epochs, val_losses, label=label, linewidth=2, 
                color=colors[i % len(colors)], marker='s', markersize=3, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss - All Trials', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    # Plot 3: All Trials - Training Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (full_name, log_info) in enumerate(all_logs.items()):
        epochs, _, train_accs, _, _ = extract_metrics(log_info['data'])
        label = log_info['trial_name']
        ax3.plot(epochs, train_accs, label=label, linewidth=2, 
                color=colors[i % len(colors)], marker='o', markersize=3, alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Training Accuracy', fontsize=12)
    ax3.set_title('Training Accuracy - All Trials', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2, loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(left=0)
    
    # Plot 4: All Trials - Validation Accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (full_name, log_info) in enumerate(all_logs.items()):
        epochs, _, _, _, val_accs = extract_metrics(log_info['data'])
        label = log_info['trial_name']
        final_acc = val_accs[-1] if val_accs else 0
        ax4.plot(epochs, val_accs, label=f"{label} (Final: {final_acc:.3f})", 
                linewidth=2, color=colors[i % len(colors)], marker='s', markersize=3, alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Validation Accuracy', fontsize=12)
    ax4.set_title('Validation Accuracy - All Trials', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=8, ncol=2, loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    ax4.set_xlim(left=0)
    
    # Plot 5: ConvNeXt vs MS-ConvNeXt Comparison (if both exist)
    if convnext_logs and msconvnext_logs:
        ax5 = fig.add_subplot(gs[2, :])
        
        # Average curves for each model type
        for model_type, logs_dict in [('ConvNeXt', convnext_logs), ('MS-ConvNeXt', msconvnext_logs)]:
            all_epochs = []
            all_val_accs = []
            
            for log_info in logs_dict.values():
                epochs, _, _, _, val_accs = extract_metrics(log_info['data'])
                all_epochs.append(epochs)
                all_val_accs.append(val_accs)
            
            # Find common epoch range
            if all_epochs:
                max_epochs = max(len(e) for e in all_epochs)
                avg_val_accs = []
                
                for epoch_idx in range(max_epochs):
                    epoch_accs = []
                    for val_acc_list in all_val_accs:
                        if epoch_idx < len(val_acc_list):
                            epoch_accs.append(val_acc_list[epoch_idx])
                    if epoch_accs:
                        avg_val_accs.append(np.mean(epoch_accs))
                
                epochs_common = list(range(1, len(avg_val_accs) + 1))
                ax5.plot(epochs_common, avg_val_accs, label=f'{model_type} (Average)', 
                        linewidth=3, marker='o', markersize=5)
        
        ax5.set_xlabel('Epoch', fontsize=12)
        ax5.set_ylabel('Validation Accuracy', fontsize=12)
        ax5.set_title('Model Comparison: ConvNeXt vs MS-ConvNeXt (Average Validation Accuracy)', 
                     fontsize=14, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)
        ax5.set_xlim(left=0)
    
    fig.suptitle('ConvNeXt Training Progress - Comprehensive Analysis', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "convnext_all_trials_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# -----------------------------
# Create Summary Statistics
# -----------------------------
def create_summary_statistics(all_logs: Dict, output_dir: str):
    """Create summary statistics table"""
    print("\n" + "="*80)
    print("Creating summary statistics...")
    print("="*80)
    
    summary_data = []
    
    for full_name, log_info in all_logs.items():
        epochs, train_losses, train_accs, val_losses, val_accs = extract_metrics(log_info['data'])
        
        if not epochs:
            continue
        
        best_val_acc = max(val_accs) if val_accs else 0
        best_val_epoch = epochs[val_accs.index(best_val_acc)] if val_accs else 0
        
        summary_data.append({
            'Trial': log_info['trial_name'],
            'Model Type': log_info['model_type'],
            'Epochs': len(epochs),
            'Final Train Loss': f"{train_losses[-1]:.4f}",
            'Final Val Loss': f"{val_losses[-1]:.4f}",
            'Final Train Acc': f"{train_accs[-1]:.4f}",
            'Final Val Acc': f"{val_accs[-1]:.4f}",
            'Best Val Acc': f"{best_val_acc:.4f}",
            'Best Val Epoch': best_val_epoch
        })
    
    # Save as JSON
    summary_path = Path(output_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for data in summary_data:
        print(f"\n{data['Trial']} ({data['Model Type']}):")
        print(f"  Epochs: {data['Epochs']}")
        print(f"  Best Val Acc: {data['Best Val Acc']} (Epoch {data['Best Val Epoch']})")
        print(f"  Final Val Acc: {data['Final Val Acc']}")

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    print("="*80)
    print("CONVNEXT TRAINING PROGRESS ANALYSIS")
    print("="*80)
    
    # Load all training logs
    all_logs = load_training_logs(SAVE_DIR, MODEL_TYPES)
    
    if not all_logs:
        print("\n❌ No training logs found!")
        print(f"   Searched in: {SAVE_DIR}")
        print(f"   Model types: {MODEL_TYPES}")
        exit(1)
    
    print(f"\n✓ Loaded {len(all_logs)} training logs")
    
    # Create visualizations
    create_individual_curves(all_logs, OUTPUT_DIR)
    create_comparative_curves(all_logs, OUTPUT_DIR)
    create_summary_statistics(all_logs, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("="*80)

