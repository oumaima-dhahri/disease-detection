# MS-ConvNeXt: Advanced Deep Learning Framework for High-Precision Wheat Disease Detection and Classification

## 🧠 Objective
Detect and classify wheat diseases from leaf images using deep learning, with a focus on multi-scale feature fusion for improved accuracy across diverse lesion patterns.

## 📊 Dataset
- **Source**: Custom wheat disease dataset
- **Total Images**: 3,745 images
- **Classes**: 12 disease categories
  - aphid (295 images)
  - army_worm (285 images)
  - black_rust (274 images)
  - brown_rust (299 images)
  - common_rust (299 images)
  - fusarium_head_blight (257 images)
  - healthy (565 images)
  - leaf_blight (291 images)
  - powdery_mildew_leaf (300 images)
  - septoria (300 images)
  - tan_spot (280 images)
  - yellow_rust (300 images)
- **Split**: 70% training, 15% validation, 15% test

## 🛠️ Method
1. **Data Preprocessing**: Image resizing, normalization, and augmentation (MixUp, CutMix)
2. **Model Training**: Multi-Scale Feature Fusion (MSF) module integrated into ConvNeXt
3. **Evaluation & Testing**: 5-fold cross-validation with comprehensive metrics

## 🧪 Models Evaluated
- **MS-ConvNeXt** (Proposed): Multi-Scale Feature Fusion with ConvNeXt backbone
- **ConvNeXt**: Baseline modernized CNN architecture
- **Hybrid CNN-ViT**: Combined CNN and Vision Transformer
- **Hybrid V2**: Enhanced hybrid architecture
- **YOLOv9+EfficientNet**: Detection-based approach

### Architecture Details
- **MS-ConvNeXt**: Parallel multi-scale branches (3×3, 5×5, 7×7 convolutions)
- **Loss Function**: Focal Loss (γ=2.7) for class imbalance handling
- **Optimizer**: AdamW with cosine annealing schedule
- **Augmentation**: MixUp (α=0.4), CutMix (α=1.0), Test-Time Augmentation (7 views)

## 📈 Results

### Overall Performance
- **MS-ConvNeXt**: 92.53% accuracy (with TTA), 92.28% weighted F1-score
- **ConvNeXt Baseline**: 91.47% accuracy, 91.32% weighted F1-score
- **Hybrid CNN-ViT**: 90.94% accuracy, 90.70% weighted F1-score
- **Hybrid V2**: 89.70% accuracy, 89.53% weighted F1-score
- **YOLOv9+EfficientNet**: 89.52% accuracy, 89.38% weighted F1-score

### Best Performing Classes
- **yellow_rust**: 100% F1-score (perfect classification)
- **fusarium_head_blight**: 98.6% F1-score
- **spetoria**: 98.8% F1-score
- **army_worm**: 98.8% F1-score

### Challenging Classes
- **tan_spot**: 65.1% F1-score (visual similarity with leaf_blight)
- **leaf_blight**: 73.9% F1-score (heterogeneous textures)

### Key Improvements
- **MSF Module**: +0.42% accuracy improvement over baseline
- **With TTA**: +1.06% total improvement (92.53% vs 91.47%)

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/oumaima-dhahri/disease-detection.git
cd disease-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
- Place your dataset in the `data/` directory
- Organize images by class folders
- Update dataset path in configuration files

### 4. Training
```bash
# Train MS-ConvNeXt model
python src/training/train_ms_convnext.py --config config/config.yaml

# Or use Jupyter notebook
jupyter notebook notebooks/training/train_MSconvnext_grid_search.ipynb
```

### 5. Evaluation
```bash
# Test trained model
python src/evaluation/test_ms_convnext.py --model_path results/models/best_model.pth

# Or use Jupyter notebook
jupyter notebook notebooks/evaluation/test_MSconvnext_grid_search.ipynb
```

### 6. Generate Visualizations
```bash
# Generate Grad-CAM visualizations
python src/evaluation/interpretability.py --model_path results/models/best_model.pth --image_path data/sample/test_images/
```

## 📁 Project Structure
```
disease-detection/
├── data/                    # Dataset (sample images)
│   └── sample/             # Sample images for each class
├── docs/                    # Documentation and images
│   └── images/             # Architecture diagrams, charts
├── src/                     # Source code
│   ├── models/             # Model architectures
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation and testing scripts
│   └── utils/              # Utility functions
├── results/                 # Results and outputs
│   ├── models/             # Saved model weights
│   ├── graphs/             # Performance charts
│   └── metrics/            # Evaluation metrics
├── notebooks/              # Jupyter notebooks
│   ├── training/           # Training notebooks
│   └── evaluation/         # Evaluation notebooks
├── README.md
└── requirements.txt
```

## 🔬 Key Features

- **Multi-Scale Feature Fusion**: Captures disease symptoms at multiple spatial scales
- **Comprehensive Benchmark**: Systematic evaluation of 5 deep learning architectures
- **Interpretability**: Grad-CAM visualizations for model explainability
- **Reproducible**: Standardized evaluation protocols with 5-fold cross-validation
- **Efficient**: Computational efficiency analysis for deployment considerations

## 📊 Performance Metrics

All models were evaluated using:
- **Accuracy**: Overall classification correctness
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to correctly identify diseased samples
- **F1-Score**: Harmonic mean of precision and recall
- **Macro Average**: Unweighted mean across all classes
- **Weighted Average**: Support-weighted mean

## 🎯 Interpretability

Grad-CAM visualizations confirm that models focus on biologically relevant lesion regions:
- Lesion boundaries and chlorotic zones are highlighted
- Multi-scale architecture enables enhanced localization
- 7×7 branch shows strong activation for extensive lesions
- 3×3 branch captures fine-grained patterns at lesion margins

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{dhahri2024msconvnext,
  title={MS-ConvNeXt: Advanced Deep Learning Framework for High-Precision Wheat Disease Detection and Classification},
  author={Dhahri, Oumaima},
  journal={Journal Name},
  year={2024}
}
```

## 👤 Author

**Oumaima Dhahri**
- Department of Data Science and Information Retrieval
- ISAMM – Institut Supérieur des Arts Multimédia de la Manouba
- Manouba, Tunisia
- Email: oumaima.dhahri@130@gmail.com
- ORCID: 0000-0000-0000-0000

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to all contributors and researchers in the agricultural AI community
- Dataset contributors and annotators
- Open-source deep learning frameworks (PyTorch, torchvision)

## 📚 References

Key references and related work are documented in the publication and `docs/` directory.

---

**Note**: This is a research project for automated wheat disease detection. For production deployment, additional validation and testing in real-world field conditions is recommended.

