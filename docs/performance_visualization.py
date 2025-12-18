#!/usr/bin/env python3
"""
Wheat Disease Detection: Performance Visualization and Comparison Charts
Generate comprehensive visualizations for model performance analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_performance_comparison_chart():
    """Create main performance comparison chart"""
    
    # Model performance data
    models = ['ConvNeXt', 'SC-ConvNeXt', 'ProtoPNet', 'Hybrid CNN-ViT', 'YOLOv9']
    accuracy = [90.93, 88.89, 70.07, 0, 0]  # 0 for incomplete models
    macro_f1 = [90.08, 88.69, 68.27, 0, 0]
    weighted_f1 = [90.34, 88.85, 69.72, 0, 0]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Chart 1: Accuracy Comparison
    colors = ['#2E8B57', '#3CB371', '#FF6B6B', '#DDA0DD', '#F0E68C']
    bars1 = ax1.bar(models, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracy):
        if acc > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=45)
    
    # Chart 2: F1-Score Comparison
    x = np.arange(len(models))
    width = 0.25
    
    bars2 = ax2.bar(x - width, macro_f1, width, label='Macro F1', color='#2E8B57', alpha=0.8)
    bars3 = ax2.bar(x, weighted_f1, width, label='Weighted F1', color='#3CB371', alpha=0.8)
    
    # Add value labels
    for bar, f1 in zip(bars2, macro_f1):
        if f1 > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{f1:.2f}%', ha='center', va='bottom', fontsize=10)
    
    for bar, f1 in zip(bars3, weighted_f1):
        if f1 > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{f1:.2f}%', ha='center', va='bottom', fontsize=10)
    
    ax2.set_title('F1-Score Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('F1-Score (%)', fontsize=14)
    ax2.set_xlabel('Models', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/performance_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_per_class_performance_chart():
    """Create per-class performance visualization"""
    
    # Per-class F1-scores for ConvNeXt (best model)
    classes = ['army_worm', 'yellow_rust', 'brown_rust', 'healthy', 'fusarium_head_blight',
               'spetoria', 'aphid', 'powdery_mildew_leaf', 'black_rust', 'common_rust',
               'leaf_blight', 'tan_spot']
    
    convnext_f1 = [100.00, 100.00, 97.30, 96.91, 96.70, 95.89, 94.55, 94.00, 90.38, 85.33, 71.91, 57.97]
    sc_convnext_f1 = [95.74, 99.05, 96.00, 94.95, 95.35, 91.89, 83.50, 88.89, 90.00, 86.84, 71.43, 70.59]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, convnext_f1, width, label='ConvNeXt', color='#2E8B57', alpha=0.8)
    bars2 = ax.bar(x + width/2, sc_convnext_f1, width, label='SC-ConvNeXt', color='#3CB371', alpha=0.8)
    
    # Add value labels
    for bar, f1 in zip(bars1, convnext_f1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{f1:.1f}%', ha='center', va='bottom', fontsize=9, rotation=90)
    
    for bar, f1 in zip(bars2, sc_convnext_f1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{f1:.1f}%', ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Color code performance levels
    for i, (conv_f1, sc_f1) in enumerate(zip(convnext_f1, sc_convnext_f1)):
        # Color based on ConvNeXt performance
        if conv_f1 >= 95:
            bars1[i].set_color('#228B22')  # Green for excellent
        elif conv_f1 >= 85:
            bars1[i].set_color('#32CD32')  # Light green for good
        elif conv_f1 >= 70:
            bars1[i].set_color('#FFD700')  # Gold for moderate
        else:
            bars1[i].set_color('#FF6347')  # Red for challenging
    
    ax.set_title('Per-Class F1-Score Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('F1-Score (%)', fontsize=14)
    ax.set_xlabel('Disease Classes', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add performance level annotations
    ax.text(0.02, 0.98, 'Performance Levels:\n🟢 ≥95%: Excellent\n🟡 85-94%: Good\n🟠 70-84%: Moderate\n🔴 <70%: Challenging', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/per_class_performance_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_sota_comparison_chart():
    """Create SOTA comparison chart"""
    
    # SOTA models and their reported accuracies
    sota_models = ['ViT-Large', 'Swin Transformer', 'DeiT', 'EfficientNet-B7', 'ResNet-152', 'DenseNet-201', 'CNN+Transformer', 'Multi-scale CNN']
    sota_accuracy_min = [92, 90, 89, 88, 85, 87, 91, 89]
    sota_accuracy_max = [95, 93, 92, 91, 89, 90, 94, 92]
    
    # Our ConvNeXt performance
    our_accuracy = 90.93
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create error bars for SOTA ranges
    y_pos = np.arange(len(sota_models))
    ax.errorbar(sota_accuracy_min, y_pos, xerr=[np.zeros(len(sota_models)), 
                np.array(sota_accuracy_max) - np.array(sota_accuracy_min)], 
                fmt='o', capsize=5, capthick=2, markersize=8, 
                label='SOTA Reported Range', color='#4169E1', alpha=0.7)
    
    # Add our ConvNeXt performance
    ax.axvline(x=our_accuracy, color='#FF4500', linestyle='--', linewidth=3, 
               label=f'Our ConvNeXt ({our_accuracy}%)')
    
    # Add performance gap annotations
    for i, (min_acc, max_acc) in enumerate(zip(sota_accuracy_min, sota_accuracy_max)):
        if our_accuracy >= min_acc and our_accuracy <= max_acc:
            ax.text(our_accuracy + 0.5, i, '🟢 Competitive', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        elif our_accuracy > max_acc:
            ax.text(our_accuracy + 0.5, i, f'🟢 +{our_accuracy-max_acc:.1f}%', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax.text(our_accuracy + 0.5, i, f'🟡 -{min_acc-our_accuracy:.1f}%', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_title('ConvNeXt vs. State-of-the-Art Models', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Accuracy (%)', fontsize=14)
    ax.set_ylabel('SOTA Models', fontsize=14)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sota_models)
    ax.set_xlim(80, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add summary statistics
    competitive_count = sum(1 for min_acc, max_acc in zip(sota_accuracy_min, sota_accuracy_max) 
                           if our_accuracy >= min_acc and our_accuracy <= max_acc)
    better_count = sum(1 for max_acc in sota_accuracy_max if our_accuracy > max_acc)
    
    ax.text(0.02, 0.98, f'Summary:\n🟢 Competitive with {competitive_count} models\n🟢 Outperforms {better_count} models\n🟡 Below {len(sota_models)-competitive_count-better_count} models', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/sota_comparison_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_efficiency_radar_chart():
    """Create efficiency radar chart"""
    
    # Categories for efficiency evaluation
    categories = ['Accuracy', 'Training Speed', 'Inference Speed', 'Memory Efficiency', 'Interpretability', 'Robustness']
    
    # Scores for each model (0-10 scale)
    convnext_scores = [9.1, 9.5, 9.5, 9.0, 3.0, 8.0]
    sc_convnext_scores = [8.9, 9.0, 9.0, 9.0, 4.0, 9.0]
    protopnet_scores = [7.0, 7.5, 7.0, 7.5, 9.5, 7.0]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Add ConvNeXt
    convnext_scores += convnext_scores[:1]
    ax.plot(angles, convnext_scores, 'o-', linewidth=2, label='ConvNeXt', color='#2E8B57')
    ax.fill(angles, convnext_scores, alpha=0.25, color='#2E8B57')
    
    # Add SC-ConvNeXt
    sc_convnext_scores += sc_convnext_scores[:1]
    ax.plot(angles, sc_convnext_scores, 'o-', linewidth=2, label='SC-ConvNeXt', color='#3CB371')
    ax.fill(angles, sc_convnext_scores, alpha=0.25, color='#3CB371')
    
    # Add ProtoPNet
    protopnet_scores += protopnet_scores[:1]
    ax.plot(angles, protopnet_scores, 'o-', linewidth=2, label='ProtoPNet', color='#FF6B6B')
    ax.fill(angles, protopnet_scores, alpha=0.25, color='#FF6B6B')
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 10)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Add title
    plt.title('Model Efficiency Radar Chart', size=16, y=1.1, fontweight='bold')
    
    # Add grid
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('docs/efficiency_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table():
    """Create a summary table visualization"""
    
    # Data for the table
    data = {
        'Model': ['ConvNeXt', 'SC-ConvNeXt', 'ProtoPNet', 'Hybrid CNN-ViT', 'YOLOv9'],
        'Accuracy (%)': [90.93, 88.89, 70.07, 'N/A', 'N/A'],
        'Macro F1 (%)': [90.08, 88.69, 68.27, 'N/A', 'N/A'],
        'Weighted F1 (%)': [90.34, 88.85, 69.72, 'N/A', 'N/A'],
        'Training Time': ['Fast', 'Fast', 'Medium', 'Slow', 'Fast'],
        'Interpretability': ['Low', 'Low', 'High', 'Medium', 'Low'],
        'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '🔄 Training', '🔄 Training']
    }
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code status
    for i in range(len(df)):
        if df.iloc[i]['Status'] == '✅ Complete':
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor('#E8F5E8')
        elif df.iloc[i]['Status'] == '🔄 Training':
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor('#FFF3CD')
    
    plt.title('Wheat Disease Detection Models - Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('docs/summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all visualization charts"""
    print("🌾 Generating Wheat Disease Detection Performance Visualizations...")
    
    # Create all charts
    create_performance_comparison_chart()
    create_per_class_performance_chart()
    create_sota_comparison_chart()
    create_efficiency_radar_chart()
    create_summary_table()
    
    print("✅ All visualization charts generated successfully!")
    print("📁 Charts saved in 'docs/' directory:")
    print("   - performance_comparison_chart.png")
    print("   - per_class_performance_chart.png")
    print("   - sota_comparison_chart.png")
    print("   - efficiency_radar_chart.png")
    print("   - summary_table.png")

if __name__ == "__main__":
    main()
