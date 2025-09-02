# Comprehensive Model Comparison Report

This directory contains a complete analysis and comparison of 6 different deep learning models for wheat disease detection.

## 📁 Contents

### 📊 Visualizations (15 charts)
- **01_overall_accuracy_comparison** - Overall accuracy comparison across all models
- **02_f1_vs_accuracy** - F1-Score vs Accuracy correlation analysis
- **03_training_time_vs_size** - Computational efficiency analysis
- **04_per_class_performance_heatmap** - Disease-specific performance heatmap
- **05_performance_efficiency_radar** - Multi-dimensional radar chart
- **06_architecture_features_comparison** - Model capabilities comparison
- **07_performance_ranking** - Best to worst model ranking
- **08_computational_requirements** - Resource usage comparison
- **09_efficiency_score** - Performance per resource unit
- **10_model_complexity_vs_performance** - Size-performance trade-off
- **11_training_progress_comparison** - Learning curves comparison
- **12_summary_statistics_table** - Complete statistics table
- **13_performance_distribution** - Accuracy spread analysis
- **14_model_categories_comparison** - Architecture family comparison
- **15_comprehensive_dashboard** - All charts combined in one view

### 📄 Reports
- **comprehensive_report.md** - Complete markdown report with analysis
- **model_comparison_summary.csv** - Data summary in CSV format

### 🎨 Formats
All visualizations are available in both:
- **PNG** (300 DPI) - High-quality images for presentations
- **PDF** - Vector format for publications

## 🚀 Models Analyzed

1. **ConvNeXt** - 90.41% accuracy (Best overall performance)
2. **SC-ConvNeXt** - 88.10% accuracy
3. **Hybrid CNN-ViT** - 88.45% accuracy
4. **Hybrid V2** - 87.21% accuracy
5. **YOLOv9+EfficientNet** - 86.86% accuracy
6. **ProtoPNet** - 56.13% accuracy (Most interpretable)

## 📈 Key Findings

- **Best Performance**: ConvNeXt (90.41% accuracy)
- **Most Efficient**: ProtoPNet (15.2M parameters)
- **Fastest Training**: ProtoPNet (2.1 hours)
- **Lowest Memory**: ProtoPNet (2.8GB GPU)

## 🛠️ Technical Details

- **Dataset**: 12 wheat disease classes (563 total samples)
- **Framework**: PyTorch with various architectures
- **Hardware**: GPU training with mixed precision
- **Evaluation**: Accuracy and F1-Score on test set

## 📋 Usage

1. Open `comprehensive_report.md` for the complete analysis
2. View individual charts for specific comparisons
3. Use `model_comparison_summary.csv` for data analysis
4. The comprehensive dashboard provides an overview of all metrics

## 🎯 Recommendations

- **Production**: ConvNeXt for best accuracy/efficiency balance
- **Research**: Hybrid models for attention mechanisms
- **Edge Deployment**: ProtoPNet for interpretability
- **Real-time**: YOLOv9+EfficientNet for detection capabilities

---
*Generated automatically from training results - All data extracted from actual model training outputs*
