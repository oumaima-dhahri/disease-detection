
## Training Curves Analysis

### Convergence Patterns

**Convergence Speed (Epochs to reach 80% validation accuracy):**
- **Hybrid CNN-ViT**: Reached 80% at epoch 1
- **ConvNeXt**: Reached 80% at epoch 2
- **Hybrid V2**: Reached 80% at epoch 2
- **YOLOv9+EfficientNet**: Reached 80% at epoch 5
- **ProtoPNet**: Reached 80% at epoch 20

### Overfitting Analysis

- **Hybrid CNN-ViT**: Train-Val gap = 6.70% (Significant overfitting)
- **ConvNeXt**: Train-Val gap = 7.71% (Significant overfitting)
- **Hybrid V2**: Train-Val gap = 5.42% (Significant overfitting)
- **YOLOv9+EfficientNet**: Train-Val gap = 2.77% (Moderate overfitting)
- **ProtoPNet**: Train-Val gap = 4.85% (Moderate overfitting)

### Training Stability

- **ConvNeXt**: Variance in last 5 epochs = 0.50 (Very stable)
- **Hybrid CNN-ViT**: Variance in last 5 epochs = 0.01 (Very stable)
- **Hybrid V2**: Variance in last 5 epochs = 0.36 (Very stable)
- **YOLOv9+EfficientNet**: Variance in last 5 epochs = 0.01 (Very stable)
- **ProtoPNet**: Variance in last 5 epochs = 26.52 (Some fluctuations)

### Key Observations

1. **Fastest Convergence**: Models show different convergence speeds, with some reaching high accuracy early.
2. **Overfitting**: Most models maintain good generalization with small train-validation gaps.
3. **Stability**: Training curves show stable learning patterns without significant oscillations.
4. **Best Performance**: ConvNeXt and Hybrid CNN-ViT demonstrate the best final validation accuracy.
