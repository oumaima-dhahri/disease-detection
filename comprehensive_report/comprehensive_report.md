
# COMPREHENSIVE MODEL COMPARISON REPORT
## Wheat Disease Detection Analysis

**Generated on:** 2025-09-02 10:27:04

### Executive Summary
This report presents a comprehensive comparison of 6 different deep learning models for wheat disease detection, 
analyzing their performance across multiple dimensions including accuracy, efficiency, and computational requirements.

### Model Performance Overview

| Model | Accuracy (%) | F1-Score (%) | Training Time (h) | Model Size (M) | GPU Memory (GB) |
|-------|-------------|-------------|------------------|---------------|----------------|
| ConvNeXt | 90.41 | 89.99 | 2.5 | 28.6 | 4.2 |
| SC-ConvNeXt | 88.10 | 87.50 | 3.1 | 32.1 | 4.8 |
| Hybrid CNN-ViT | 88.45 | 88.35 | 4.2 | 45.8 | 6.1 |
| Hybrid V2 | 87.21 | 87.22 | 3.8 | 38.9 | 5.3 |
| YOLOv9+EfficientNet | 86.86 | 86.23 | 5.5 | 52.3 | 7.2 |
| ProtoPNet | 56.13 | 53.55 | 2.1 | 15.2 | 2.8 |


### Key Findings

1. **Best Overall Performance**: ConvNeXt achieved the highest accuracy at 90.41%
2. **Most Efficient**: ProtoPNet has the smallest model size at 15.2M parameters
3. **Fastest Training**: ProtoPNet completed training in 2.1 hours
4. **Lowest Memory Usage**: ProtoPNet requires only 2.8GB GPU memory

### Performance Rankings

1. **ConvNeXt** - 90.41% accuracy
2. **Hybrid CNN-ViT** - 88.45% accuracy  
3. **SC-ConvNeXt** - 88.10% accuracy
4. **Hybrid V2** - 87.21% accuracy
5. **YOLOv9+EfficientNet** - 86.86% accuracy
6. **ProtoPNet** - 56.13% accuracy

### Model Categories Analysis

- **ConvNeXt Family**: Average accuracy of 89.25%
- **Hybrid Models**: Average accuracy of 87.83%
- **Detection Models**: Average accuracy of 86.86%
- **Interpretable Models**: Average accuracy of 56.13%

### Recommendations

1. **For Production**: ConvNeXt offers the best balance of accuracy and efficiency
2. **For Research**: Hybrid models show promising results with attention mechanisms
3. **For Edge Deployment**: ProtoPNet provides interpretability with reasonable performance
4. **For Real-time Applications**: YOLOv9+EfficientNet offers detection capabilities

### Technical Details

- **Dataset**: 12 wheat disease classes with 563 total samples
- **Evaluation Metric**: Accuracy and F1-Score on test set
- **Hardware**: GPU training with mixed precision
- **Framework**: PyTorch with various architectures

### Generated Visualizations

This report includes 15 comprehensive visualizations:
1. Overall Accuracy Comparison
2. F1-Score vs Accuracy Correlation
3. Training Time vs Model Size
4. Per-Class Performance Heatmap
5. Performance vs Efficiency Radar Chart
6. Architecture Features Comparison
7. Performance Ranking
8. Computational Requirements
9. Efficiency Score
10. Model Complexity vs Performance
11. Training Progress Comparison
12. Summary Statistics Table
13. Performance Distribution
14. Model Categories Comparison
15. Comprehensive Dashboard

All visualizations are available in both PNG (300 DPI) and PDF formats in the 'comprehensive_report/' directory.
