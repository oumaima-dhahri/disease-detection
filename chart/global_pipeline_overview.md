# Wheat Disease Detection: Global Pipeline

## Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           WHEAT DISEASE DETECTION PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────────┘

1. DATA PREPROCESSING
   ├── Raw Images (12 classes: aphid, army_worm, black_rust, brown_rust, etc.)
   ├── Quality Validation & Corruption Detection
   ├── Stratified Split (Train: 70%, Val: 15%, Test: 15%)
   ├── Data Augmentation (Flip, Rotate, Color Jitter, Random Crop)
   ├── Normalization (ImageNet stats: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   └── DataLoader Setup (Batch=32, Workers=4, Persistent=True)

2. MODEL ARCHITECTURE
   ├── ConvNeXt (ConvNeXt-Tiny, 28M params, 4 stages)
   ├── SC-ConvNeXt (Structured sparsity, λ=0.01)
   ├── Hybrid CNN-ViT (ResNet-50 + ViT-Base)
   ├── Hybrid V2 (Enhanced fusion, adaptive attention)
   ├── YOLOv9+EfficientNet (EfficientNet-B3 + PANet)
   └── ProtoPNet (VGG-19 + 10 prototypes/class)

3. TRAINING
   ├── Mixed Precision Training (FP16 forward, FP32 gradients)
   ├── Progressive Training Protocol (3 phases)
   ├── Early Stopping (Patience=5 epochs)
   ├── Checkpointing (Every 5 epochs)
   ├── Loss Optimization (CrossEntropy + Auxiliary losses)
   └── Validation Monitoring (TensorBoard logging)

4. EVALUATION
   ├── Basic Metrics
   │   ├── Accuracy, Precision, Recall, F1-Score
   │   ├── Macro Average (handles class imbalance)
   │   └── Weighted Average (by class frequency)
   ├── Advanced Metrics
   │   ├── AUC-ROC (Area Under ROC Curve)
   │   ├── AUC-PR (Precision-Recall curve)
   │   ├── Cohen's Kappa (agreement measure)
   │   └── Matthews Correlation Coefficient (MCC)
   ├── Statistical Testing
   │   ├── Confidence Intervals (95% CI, 1000 bootstrap samples)
   │   ├── Wilcoxon Signed-Rank Test
   │   └── Effect Size (Cohen's d)
   └── Robustness Analysis
       ├── Confusion Matrix Analysis
       ├── Cross-Fold Consistency
       └── Outlier Detection

5. INTERPRETABILITY
   ├── Gradient-Based Explanations
   │   ├── Grad-CAM (Gradient-weighted class activation mapping)
   │   ├── Integrated Gradients
   │   └── Saliency Maps
   ├── Prototype Analysis (ProtoPNet specific)
   │   ├── Prototype Visualization
   │   ├── Prototype Localization
   │   └── Reasoning Transparency
   └── Feature Visualization
       ├── Decision Boundary Analysis
       └── Feature Space Visualization

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TECHNICAL INFRASTRUCTURE                          │
└─────────────────────────────────────────────────────────────────────────────────┘

HARDWARE CONFIGURATION:
├── GPU: NVIDIA Tesla T4 (16GB VRAM)
├── CPU: Intel Xeon E5-2686 v4 (2.3 GHz, 16 cores)
├── RAM: 64GB DDR4 ECC memory
├── Storage: 500GB SSD
└── Platform: Kaggle Notebooks

SOFTWARE STACK:
├── Deep Learning: PyTorch 1.12.1 + CUDA 11.8
├── Computer Vision: OpenCV 4.6.0, PIL 9.2.0
├── Scientific Computing: NumPy 1.21.6, SciPy 1.9.1
├── Data Processing: Pandas 1.4.4, Scikit-learn 1.1.1
└── Visualization: Matplotlib 3.5.3, Seaborn 0.11.2

REPRODUCIBILITY MEASURES:
├── Fixed Random Seeds (PyTorch: 42, NumPy: 42, Python: 42)
├── CUDA Determinism (torch.backends.cudnn.deterministic = True)
├── Algorithm Determinism (torch.use_deterministic_algorithms(True))
└── Fixed Data Loading (consistent worker initialization)

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                PIPELINE FLOW                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

Raw Data → Preprocessing → Model Selection → Training → Evaluation → Interpretability
    ↓           ↓              ↓             ↓          ↓            ↓
  Images    Clean Data    Architecture    Trained    Metrics    Explanations
  12 Classes  Augmented    6 Models       Models     Advanced    Visualizations

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT DELIVERABLES                               │
└─────────────────────────────────────────────────────────────────────────────────┘

1. Trained Models (6 architectures)
2. Performance Metrics (Comprehensive comparison)
3. Visualizations (15+ charts and diagrams)
4. Statistical Analysis (Significance testing)
5. Interpretability Reports (Grad-CAM, prototypes)
6. Technical Documentation (Complete methodology)
7. Reproducible Code (All scripts and configurations)

