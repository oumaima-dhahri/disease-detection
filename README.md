# Wheat Disease Detection with ConvNeXt and Explainable AI

This project provides tools to test a trained ConvNeXt model for wheat disease detection and visualize model predictions using multiple explainability heatmaps.

## Features
- **Test your trained ConvNeXt model** on new images.
- **Visualize model predictions** with:
  - Grad-CAM (class activation map)
  - Saliency Map (input gradient)
  - ViT-style Attention Map (mean of last feature map channels)
- **Easy to extend** with more explainability methods (e.g., Guided Backprop, SmoothGrad).

## Setup
1. **Clone or download this repository.**
2. **Install dependencies:**
   ```bash
   pip install torch torchvision pillow numpy matplotlib opencv-python
   ```
3. **Place your trained model** at:
   ```
   saved_models_and_data/best_convnext_model.pth
   ```
4. **Add test images** to the `test_images` folder.

## Usage
Run the test script to visualize predictions and heatmaps:
```bash
python test_convnext.py
```
For each image, the script will display:
- The original image
- Grad-CAM heatmap
- Grad-CAM overlay
- Saliency overlay
- ViT-style attention overlay

## Heatmap Methods Explained
- **Grad-CAM:** Highlights regions most important for the model's prediction.
- **Saliency Map:** Shows pixel-wise importance using input gradients.
- **ViT-style Attention Map:** Approximates attention by averaging the last feature map channels (for ConvNeXt).

## How to Extend
You can add more explainability methods, such as:
- **Guided Backpropagation**
- **SmoothGrad**
- **Integrated Gradients**
- **Occlusion Sensitivity**

See the code comments and docstrings in `test_convnext.py` for templates and instructions.

## Citation
If you use this code for research, please cite the original ConvNeXt and explainability method papers.

## License
[MIT License](LICENSE)

---
**Author:** [Your Name]

**Date:** [Date] 