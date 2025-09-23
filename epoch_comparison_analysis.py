#!/usr/bin/env python3
"""
Comprehensive Comparison Analysis: Epoch 10 vs Epoch 20 Training Results
=======================================================================

This script analyzes and compares the training results from epoch 10 and epoch 20
for all models in the wheat disease detection project.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EpochComparisonAnalyzer:
    """Analyzes and compares training results between epoch 10 and epoch 20."""
    
    def __init__(self):
        self.models = [
            'ConvNeXt',
            'Hybrid CNN-ViT', 
            'Hybrid V2',
            'ProtoPNet',
            'SC-ConvNeXt',
            'YOLOv9 + EfficientNet B3'
        ]
        
        # Epoch 10 Results
        self.epoch10_results = {
            'ConvNeXt': {'accuracy': 0.9041, 'f1_macro': 0.8955, 'f1_weighted': 0.8999},
            'Hybrid CNN-ViT': {'accuracy': 0.8845, 'f1_macro': 0.8762, 'f1_weighted': 0.8835},
            'Hybrid V2': {'accuracy': 0.8721, 'f1_macro': 0.8698, 'f1_weighted': 0.8722},
            'ProtoPNet': {'accuracy': 0.5613, 'f1_macro': 0.5323, 'f1_weighted': 0.5355},
            'SC-ConvNeXt': {'accuracy': 0.8810, 'f1_macro': 0.8716, 'f1_weighted': 0.8750},
            'YOLOv9 + EfficientNet B3': {'accuracy': 0.8561, 'f1_macro': 0.8373, 'f1_weighted': 0.8481}
        }
        
        # Epoch 20 Results
        self.epoch20_results = {
            'ConvNeXt': {'accuracy': 0.9147, 'f1_macro': 0.9085, 'f1_weighted': 0.9132},
            'Hybrid CNN-ViT': {'accuracy': 0.9165, 'f1_macro': 0.9105, 'f1_weighted': 0.9156},
            'Hybrid V2': {'accuracy': 0.6998, 'f1_macro': 0.7086, 'f1_weighted': 0.7084},
            'ProtoPNet': {'accuracy': 0.6998, 'f1_macro': 0.7086, 'f1_weighted': 0.7084},
            'SC-ConvNeXt': {'accuracy': 0.9147, 'f1_macro': 0.9081, 'f1_weighted': 0.9142},
            'YOLOv9 + EfficientNet B3': {'accuracy': 0.8952, 'f1_macro': 0.8889, 'f1_weighted': 0.8938}
        }
        
        # Training time data (in minutes)
        self.training_times = {
            'Epoch 10': {
                'ConvNeXt': 9.7,  # ~10 minutes for 10 epochs
                'Hybrid CNN-ViT': 8.3,  # ~8 minutes for 10 epochs  
                'Hybrid V2': 10.9,  # ~11 minutes for 10 epochs
                'ProtoPNet': 16.9,  # ~17 minutes for 10 epochs
                'SC-ConvNeXt': 8.9,  # ~9 minutes for 10 epochs
                'YOLOv9 + EfficientNet B3': 30.0  # 30 minutes for 10 epochs
            },
            'Epoch 20': {
                'ConvNeXt': 15.0,  # Early stopping at epoch 15
                'Hybrid CNN-ViT': 16.0,  # Early stopping at epoch 16
                'Hybrid V2': 19.0,  # Early stopping at epoch 19
                'ProtoPNet': 36.0,  # Full 20 epochs
                'SC-ConvNeXt': 17.4,  # Full 20 epochs
                'YOLOv9 + EfficientNet B3': 60.1  # Full 20 epochs
            }
        }
    
    def create_comparison_dataframe(self):
        """Create a comprehensive comparison dataframe."""
        comparison_data = []
        
        for model in self.models:
            epoch10_acc = self.epoch10_results[model]['accuracy']
            epoch20_acc = self.epoch20_results[model]['accuracy']
            improvement = epoch20_acc - epoch10_acc
            improvement_pct = (improvement / epoch10_acc) * 100
            
            epoch10_f1 = self.epoch10_results[model]['f1_weighted']
            epoch20_f1 = self.epoch20_results[model]['f1_weighted']
            f1_improvement = epoch20_f1 - epoch10_f1
            
            epoch10_time = self.training_times['Epoch 10'][model]
            epoch20_time = self.training_times['Epoch 20'][model]
            time_increase = epoch20_time - epoch10_time
            
            comparison_data.append({
                'Model': model,
                'Epoch 10 Accuracy': epoch10_acc,
                'Epoch 20 Accuracy': epoch20_acc,
                'Accuracy Improvement': improvement,
                'Accuracy Improvement %': improvement_pct,
                'Epoch 10 F1-Score': epoch10_f1,
                'Epoch 20 F1-Score': epoch20_f1,
                'F1-Score Improvement': f1_improvement,
                'Epoch 10 Time (min)': epoch10_time,
                'Epoch 20 Time (min)': epoch20_time,
                'Time Increase (min)': time_increase,
                'Efficiency Score': epoch20_acc / epoch20_time  # Accuracy per minute
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_accuracy_comparison(self, df, save_path='epoch_comparison_accuracy.png'):
        """Plot accuracy comparison between epoch 10 and 20."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Accuracy comparison bar plot
        x = np.arange(len(self.models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, df['Epoch 10 Accuracy'], width, 
                       label='Epoch 10', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, df['Epoch 20 Accuracy'], width,
                       label='Epoch 20', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison: Epoch 10 vs Epoch 20')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Improvement percentage
        colors = ['green' if x > 0 else 'red' for x in df['Accuracy Improvement %']]
        bars3 = ax2.bar(x, df['Accuracy Improvement %'], color=colors, alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Accuracy Improvement (%)')
        ax2.set_title('Accuracy Improvement from Epoch 10 to 20')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_efficiency_analysis(self, df, save_path='epoch_comparison_efficiency.png'):
        """Plot efficiency analysis comparing accuracy vs training time."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Accuracy vs Training Time scatter plot
        ax1.scatter(df['Epoch 10 Time (min)'], df['Epoch 10 Accuracy'], 
                   s=100, alpha=0.7, label='Epoch 10', color='skyblue')
        ax1.scatter(df['Epoch 20 Time (min)'], df['Epoch 20 Accuracy'], 
                   s=100, alpha=0.7, label='Epoch 20', color='lightcoral')
        
        # Add model labels
        for i, model in enumerate(self.models):
            ax1.annotate(model, (df['Epoch 10 Time (min)'].iloc[i], df['Epoch 10 Accuracy'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
            ax1.annotate(model, (df['Epoch 20 Time (min)'].iloc[i], df['Epoch 20 Accuracy'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax1.set_xlabel('Training Time (minutes)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Training Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency Score comparison
        x = np.arange(len(self.models))
        bars = ax2.bar(x, df['Efficiency Score'], alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Efficiency Score (Accuracy/Minute)')
        ax2.set_title('Training Efficiency Comparison (Epoch 20)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_f1_score_comparison(self, df, save_path='epoch_comparison_f1.png'):
        """Plot F1-score comparison."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        x = np.arange(len(self.models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['Epoch 10 F1-Score'], width,
                      label='Epoch 10', alpha=0.8, color='lightblue')
        bars2 = ax.bar(x + width/2, df['Epoch 20 F1-Score'], width,
                      label='Epoch 20', alpha=0.8, color='orange')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('F1-Score (Weighted)')
        ax.set_title('F1-Score Comparison: Epoch 10 vs Epoch 20')
        ax.set_xticks(x)
        ax.set_xticklabels(self.models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_summary_table(self, df, save_path='epoch_comparison_summary.csv'):
        """Create and save a summary table."""
        summary_df = df.copy()
        
        # Round numerical columns
        numerical_cols = ['Epoch 10 Accuracy', 'Epoch 20 Accuracy', 'Accuracy Improvement',
                         'Accuracy Improvement %', 'Epoch 10 F1-Score', 'Epoch 20 F1-Score',
                         'F1-Score Improvement', 'Efficiency Score']
        
        for col in numerical_cols:
            summary_df[col] = summary_df[col].round(4)
        
        summary_df.to_csv(save_path, index=False)
        return summary_df
    
    def generate_comprehensive_report(self, df):
        """Generate a comprehensive analysis report."""
        print("="*80)
        print("COMPREHENSIVE EPOCH COMPARISON ANALYSIS")
        print("="*80)
        print()
        
        # Overall statistics
        print("📊 OVERALL STATISTICS")
        print("-" * 40)
        print(f"Average Accuracy Improvement: {df['Accuracy Improvement'].mean():.4f}")
        print(f"Best Accuracy Improvement: {df['Accuracy Improvement'].max():.4f} ({df.loc[df['Accuracy Improvement'].idxmax(), 'Model']})")
        print(f"Worst Accuracy Improvement: {df['Accuracy Improvement'].min():.4f} ({df.loc[df['Accuracy Improvement'].idxmin(), 'Model']})")
        print(f"Average Training Time Increase: {df['Time Increase (min)'].mean():.1f} minutes")
        print()
        
        # Model rankings
        print("🏆 MODEL RANKINGS (Epoch 20)")
        print("-" * 40)
        accuracy_ranking = df.sort_values('Epoch 20 Accuracy', ascending=False)
        print("By Accuracy:")
        for i, (_, row) in enumerate(accuracy_ranking.iterrows(), 1):
            print(f"{i}. {row['Model']}: {row['Epoch 20 Accuracy']:.4f}")
        print()
        
        efficiency_ranking = df.sort_values('Efficiency Score', ascending=False)
        print("By Efficiency (Accuracy/Minute):")
        for i, (_, row) in enumerate(efficiency_ranking.iterrows(), 1):
            print(f"{i}. {row['Model']}: {row['Efficiency Score']:.4f}")
        print()
        
        # Key insights
        print("🔍 KEY INSIGHTS")
        print("-" * 40)
        
        # Best performers
        best_accuracy = df.loc[df['Epoch 20 Accuracy'].idxmax()]
        best_efficiency = df.loc[df['Efficiency Score'].idxmax()]
        best_improvement = df.loc[df['Accuracy Improvement'].idxmax()]
        
        print(f"• Best Overall Accuracy: {best_accuracy['Model']} ({best_accuracy['Epoch 20 Accuracy']:.4f})")
        print(f"• Most Efficient Training: {best_efficiency['Model']} ({best_efficiency['Efficiency Score']:.4f} acc/min)")
        print(f"• Biggest Improvement: {best_improvement['Model']} (+{best_improvement['Accuracy Improvement %']:.1f}%)")
        
        # Models that benefited most from longer training
        benefited_models = df[df['Accuracy Improvement'] > 0.01].sort_values('Accuracy Improvement', ascending=False)
        if len(benefited_models) > 0:
            print(f"• Models benefiting most from longer training: {', '.join(benefited_models['Model'].tolist())}")
        
        # Models with diminishing returns
        diminishing_models = df[df['Accuracy Improvement'] < 0].sort_values('Accuracy Improvement')
        if len(diminishing_models) > 0:
            print(f"• Models with diminishing returns: {', '.join(diminishing_models['Model'].tolist())}")
        
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 40)
        print("• For production deployment: Focus on ConvNeXt, Hybrid CNN-ViT, or SC-ConvNeXt")
        print("• For resource-constrained environments: Consider SC-ConvNeXt for best efficiency")
        print("• ProtoPNet shows significant improvement potential with longer training")
        print("• Early stopping is effective for most models to prevent overfitting")
        print()
        
        return df
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 Starting Epoch Comparison Analysis...")
        
        # Create comparison dataframe
        df = self.create_comparison_dataframe()
        
        # Generate visualizations
        print("📈 Generating visualizations...")
        self.plot_accuracy_comparison(df)
        self.plot_efficiency_analysis(df)
        self.plot_f1_score_comparison(df)
        
        # Create summary table
        print("📋 Creating summary table...")
        summary_df = self.create_summary_table(df)
        
        # Generate comprehensive report
        print("📊 Generating comprehensive report...")
        self.generate_comprehensive_report(df)
        
        print("✅ Analysis complete! Check the generated files:")
        print("   - epoch_comparison_accuracy.png")
        print("   - epoch_comparison_efficiency.png") 
        print("   - epoch_comparison_f1.png")
        print("   - epoch_comparison_summary.csv")
        
        return df

def main():
    """Main function to run the analysis."""
    analyzer = EpochComparisonAnalyzer()
    results_df = analyzer.run_complete_analysis()
    return results_df

if __name__ == "__main__":
    results = main()
