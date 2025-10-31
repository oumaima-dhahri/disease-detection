# YOLOv9 Hybrid Model Test Script

This comprehensive test script (`test_yolov9_hybrid.py`) is designed to evaluate and test the trained YOLOv9 hybrid model for wheat disease detection. It provides extensive evaluation metrics, visualizations, and research reporting capabilities.

## Features

### 🎯 **Model Evaluation**
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Per-Class Performance**: Individual class analysis with AUC scores
- **Confusion Matrix**: Visual representation of classification performance
- **ROC Curves**: Receiver Operating Characteristic analysis for all classes

### 🔍 **Visualization Tools**
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Saliency Maps**: Input gradient-based attention visualization
- **Attention Maps**: Feature-based attention visualization
- **Overlay Visualizations**: Heatmaps overlaid on original images

### 📊 **Research Reporting**
- **JSON Reports**: Structured data for further analysis
- **Performance Metrics**: Comprehensive evaluation summary
- **Model Architecture**: Detailed model information
- **Dataset Information**: Configuration and statistics

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r requirements_test.txt
```

### 2. Model Requirements
- **Trained Model**: Ensure you have a trained model file at `../saved_models_and_data/best_hybrid_model.pth`
- **Dataset**: The script expects the dataset structure from the training script
- **YOLOv9 Weights**: `yolov9c.pt` should be available (will be downloaded automatically)

### 3. Directory Structure
```
disease-detection/
├── dataset/                    # Original dataset
├── dataset_split/             # Train/val/test splits
├── test_images/               # Individual test images
├── saved_models_and_data/     # Model files and results
└── test_yolov9_hybrid.py     # This test script
```

## Usage

### Basic Usage
```bash
python test_yolov9_hybrid.py
```

### What the Script Does

1. **Model Loading**: Loads the trained hybrid YOLOv9 + EfficientNet model
2. **Single Image Testing**: Tests individual images from `test_images/` directory
3. **Dataset Evaluation**: Evaluates performance on the test dataset (if available)
4. **Visualization Generation**: Creates comprehensive visualizations for each test image
5. **Metrics Calculation**: Computes all performance metrics
6. **Report Generation**: Saves detailed reports in JSON format

## Output Files

The script generates the following files in `../saved_models_and_data/evaluation_results/`:

### 📈 **Performance Metrics**
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curves.png` - ROC curves for all classes
- `classification_report.json` - Detailed classification report
- `per_class_metrics.json` - Per-class performance metrics

### 🖼️ **Visualizations**
- `visualization_[image_name].png` - Comprehensive visualization for each test image
- `single_image_results.json` - Summary of single image testing results

### 📋 **Research Reports**
- `research_report.json` - Complete research report with model information

## Understanding the Outputs

### 1. **Confusion Matrix**
- Shows true vs. predicted labels
- Diagonal elements represent correct predictions
- Off-diagonal elements show misclassifications

### 2. **ROC Curves**
- Each class gets its own curve
- Higher AUC (Area Under Curve) = better performance
- Random classifier baseline at 0.5

### 3. **Grad-CAM Visualizations**
- **Original Image**: Input image
- **Grad-CAM Heatmap**: Raw attention heatmap
- **Grad-CAM Overlay**: Heatmap overlaid on image
- **Saliency Map**: Input gradient-based attention
- **Attention Map**: Feature-based attention

### 4. **Performance Metrics**
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Customization

### Modify Model Path
```python
MODEL_PATH = '../saved_models_and_data/your_model.pth'
```

### Change Image Size
```python
IMAGE_SIZE = (224, 224)  # Adjust as needed
```

### Modify Batch Size
```python
BATCH_SIZE = 16  # Adjust based on your GPU memory
```

### Add Custom Metrics
The script is modular - you can easily add custom evaluation functions in the main section.

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure the model file exists at the specified path
   - Check if you've trained the model first using `train_yolo_v9.py`

2. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE`
   - Use CPU by modifying device selection

3. **Import Errors**
   - Install missing dependencies: `pip install -r requirements_test.txt`
   - Check Python version compatibility

4. **No Test Images**
   - Place test images in the `test_images/` directory
   - Supported formats: PNG, JPG, JPEG, JFIF

### Performance Tips

- **GPU Usage**: The script automatically detects and uses CUDA if available
- **Memory Management**: Large images may require reducing batch size
- **Visualization Quality**: High DPI saves (300 DPI) for publication-quality figures

## Research Applications

This test script is designed for research purposes and provides:

- **Reproducible Results**: All metrics are saved in structured JSON format
- **Publication-Ready Figures**: High-quality visualizations suitable for papers
- **Comprehensive Analysis**: Multiple evaluation perspectives for thorough research
- **Data Export**: Easy integration with other analysis tools

## Integration with Training

The test script is designed to work seamlessly with the training script:

1. **Train Model**: Use `train_yolo_v9.py` to train your model
2. **Test Model**: Use this script to evaluate performance
3. **Iterate**: Use results to improve model architecture or training parameters

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Ensure the model file exists and matches the expected architecture
4. Check console output for specific error messages

## License

This script is part of the wheat disease detection project and follows the same licensing terms.

