## State of the Art in Plant Disease and Pest Detection from Leaf Images

### Overview
Automated detection of crop diseases and pests from leaf images has progressed rapidly with deep learning. Classical feature-based methods (e.g., SIFT/HOG + SVM) have largely been superseded by convolutional neural networks (CNNs), which capture hierarchical texture and shape cues. Recent trends incorporate attention mechanisms and transformer architectures to model long-range dependencies, improve robustness to field conditions, and enable explainability. Data-efficient training strategies and transfer learning from large-scale natural image corpora are standard practice to overcome domain data scarcity.

### Historical Development of the Field

Before deep learning, hand-crafted features (e.g., SIFT, HOG) paired with SVM or Random Forest classifiers performed well on controlled datasets but struggled with real-world variability—non-uniform lighting, cluttered backgrounds, and growth-stage differences (Khan et al., 2024). The shift to CNNs marked a turning point: early work using AlexNet and ResNet-style architectures trained via transfer learning from ImageNet substantially improved accuracy by learning hierarchical texture and shape features relevant to lesions. Long et al. (2022) reported accuracy ranges of 85–92% on mixed glasshouse and field wheat disease datasets, demonstrating the effectiveness of transfer learning approaches (Qian et al., 2025; Long et al., 2022). Subsequently, researchers focused on efficiency: EfficientNet and lightweight CNNs (MobileNet, ShuffleNet) balanced accuracy and computational cost, enabling mobile and edge deployment for in-field scouting applications. Ennadifi et al. (2020) showed that lightweight architectures could achieve competitive performance while maintaining deployment feasibility on resource-constrained devices (Qian et al., 2024; Qian et al., 2025; Ennadifi et al., 2020). To address attention drift and background bias, channel and spatial attention mechanisms (e.g., SE blocks, CBAM) were integrated to improve focus on symptomatic regions, especially in field imagery. Nadeem et al. (2023) achieved 95.66% accuracy using modified soft attention networks, representing a significant advancement over baseline CNNs (Nadeem et al., 2023). More recently, transformer architectures—particularly Vision Transformers (ViT) and Swin Transformers with shifted windows—introduced global context modeling through self-attention, offering a favorable accuracy–compute trade-off and improved robustness to complex scenes. Khubaib et al. (2025) reported exceptional accuracy of 99.3% using Swin Transformers, demonstrating the power of global context modeling for wheat disease detection (Kumar et al., 2023; Khubaib et al., 2025). In parallel, explainability became a practical requirement: Grad-CAM and LIME became standard tools to validate that models attend to lesions rather than confounders, supporting agronomic adoption and error analysis (Dutta et al., 2021; Sneha et al., 2023). Finally, data-efficient learning strategies emerged—advanced augmentation, class rebalancing, and semi/self-supervised pretraining reduced reliance on large labeled datasets, while domain adaptation methods addressed cross-farm and cross-cultivar generalization challenges (Dong et al., 2024; Khubaib et al., 2025; Khan et al., 2024).

### Contemporary Deep Learning Architectures for Plant Disease Detection

This section reviews recent deep learning architectures that form the foundation of modern plant disease detection systems, examining their performance characteristics, efficiency profiles, and deployment considerations.

CNN backbones—particularly ResNet, DenseNet, Inception, and EfficientNet—remain strong baselines for disease classification, often fine-tuned from ImageNet pretrained weights. Long et al. (2022) demonstrated that ResNet architectures achieve 85–92% accuracy on mixed glasshouse and field wheat disease datasets, establishing a solid foundation for transfer learning approaches. Among these architectures, EfficientNet variants provide a particularly favorable accuracy–efficiency trade-off for embedded and edge scenarios, achieving 88–93% accuracy while maintaining computational efficiency, making them well-suited for deployment in resource-constrained agricultural settings (Qian et al., 2025; Long et al., 2022; Sneha et al., 2023). For applications requiring real-time inference on mobile devices or drones, lightweight CNNs such as MobileNet and ShuffleNet offer substantial reductions in computational cost and memory footprint, though with moderate accuracy trade-offs compared to heavier backbones, typically achieving 82–88% accuracy; Ennadifi et al. (2020) reported competitive performance with mobile architectures achieving 85–87% accuracy while maintaining deployment feasibility (Ennadifi et al., 2020; Qian et al., 2024). To address the challenge of background clutter and variable illumination in field imagery, attention mechanisms—including channel and spatial attention modules such as SE blocks and CBAM—have been integrated into CNN architectures to improve focus on lesion regions, yielding consistent performance gains especially under cluttered backgrounds. Nadeem et al. (2023) demonstrated that modified soft attention mechanisms can push wheat disease classification accuracy to 95.66%, representing a significant improvement over baseline CNNs (Nadeem et al., 2023). In the latest studies, transformer-based architectures have emerged as powerful alternatives: Vision Transformers (ViT) and Swin Transformers leverage self-attention to capture global context, with Swin's shifted windowing mechanism providing a favorable balance between accuracy and computational efficiency that enables scalable training and improved robustness to complex field scenes. Khubaib et al. (2025) reported exceptional performance of 99.3% accuracy using Swin Transformers on wheat disease detection tasks, capitalizing on global context modeling to handle complex symptom distributions (Khubaib et al., 2025; Kumar et al., 2023). Notably, hybrid CNN–Transformer models that combine local feature extraction with global context modeling often outperform pure CNNs on complex field images, achieving 88–95% accuracy, with Yang et al. (2023) and Sneha et al. (2023) reporting competitive results of 88–92% accuracy through hybrid architectures. In parallel, explainability has become integral to model design: Grad-CAM and LIME are widely used to validate that models attend to symptomatic regions rather than background confounders, supporting agronomic trust and adoption. Sneha et al. (2023) integrated Grad-CAM and LIME explanations into their multi-model analysis, demonstrating that explainability techniques enhance model transparency and facilitate error diagnosis (Sneha et al., 2023; Dutta et al., 2021). Some recent works go further by integrating attention maps directly into the training objective for lesion-aware learning, ensuring that high-confidence predictions align with precise lesion localization.

#### Modernized CNN Architectures

Contemporary approaches have modernized traditional CNNs by incorporating transformer-inspired design principles while maintaining computational efficiency.

Modernized CNN architectures such as ConvNeXt have emerged, combining the efficiency of CNNs with design principles inspired by transformers. ConvNeXt achieves competitive performance (often 88–92% accuracy) while maintaining the computational advantages of convolutional architectures, making it an attractive alternative to transformer-based models for scenarios requiring both high accuracy and efficiency. These architectures modernize traditional CNNs through improved normalization strategies (LayerNorm), activation functions (GELU), and macro design choices (inverted bottlenecks, larger kernel sizes), achieving performance comparable to Swin Transformers with lower computational overhead (Liu et al., 2022). Attention-augmented variants such as SC-ConvNeXt (Self-Calibrated ConvNeXt) integrate channel and spatial attention mechanisms into the ConvNeXt architecture, achieving 88–92% accuracy with improved focus on lesion regions, demonstrating that modernized CNNs can effectively compete with transformer-based approaches while maintaining deployment advantages.

#### Comparative Architecture Summary

The following table provides a comparative overview of major architecture categories, their performance characteristics, efficiency profiles, and deployment contexts:

**Markdown Table (for documentation):**

| Architecture | Accuracy Range | Efficiency | Interpretability | Best Use Case |
|--------------|---------------|-----------|------------------|---------------|
| **CNN Baselines** (ResNet, EfficientNet) | 85–93% | High (0.4–4B FLOPs) | Low (post-hoc only) | General baseline, edge deployment |
| **Lightweight CNNs** (MobileNet, ShuffleNet) | 82–88% | Very High (<1B FLOPs) | Low (post-hoc only) | Mobile devices, real-time inference |
| **Attention-Augmented CNNs** (SE, CBAM) | 90–96% | Medium (1–3B FLOPs) | Medium (attention maps) | Cluttered backgrounds, field imagery |
| **Transformers** (ViT, Swin) | 95–99% | Medium–High (2–5B FLOPs) | Medium (attention maps) | High-accuracy research, complex scenes |
| **Hybrid CNN–Transformer** | 88–95% | Medium (3–5B FLOPs) | Medium (attention maps) | Balanced accuracy and context |
| **Modernized CNNs** (ConvNeXt) | 88–92% | High (1–3B FLOPs) | Low (post-hoc only) | Balanced accuracy and efficiency |
| **Prototype-Based** (ProtoPNet) | 65–75% | High (1–2B FLOPs) | Very High (built-in) | Regulatory compliance, interpretability |
| **Detection Models** (YOLO, RetinaNet) | 80–90% mAP | Medium (2–4B FLOPs) | Low (post-hoc only) | Lesion localization, severity estimation |

**Word Table Format (copy and paste into Word, then select text and use Insert > Table > Convert Text to Table):**

```
Architecture	Accuracy Range	Efficiency	Interpretability	Best Use Case
CNN Baselines (ResNet, EfficientNet)	85–93%	High (0.4–4B FLOPs)	Low (post-hoc only)	General baseline, edge deployment
Lightweight CNNs (MobileNet, ShuffleNet)	82–88%	Very High (<1B FLOPs)	Low (post-hoc only)	Mobile devices, real-time inference
Attention-Augmented CNNs (SE, CBAM)	90–96%	Medium (1–3B FLOPs)	Medium (attention maps)	Cluttered backgrounds, field imagery
Transformers (ViT, Swin)	95–99%	Medium–High (2–5B FLOPs)	Medium (attention maps)	High-accuracy research, complex scenes
Hybrid CNN–Transformer	88–95%	Medium (3–5B FLOPs)	Medium (attention maps)	Balanced accuracy and context
Modernized CNNs (ConvNeXt)	88–92%	High (1–3B FLOPs)	Low (post-hoc only)	Balanced accuracy and efficiency
Prototype-Based (ProtoPNet)	65–75%	High (1–2B FLOPs)	Very High (built-in)	Regulatory compliance, interpretability
Detection Models (YOLO, RetinaNet)	80–90% mAP	Medium (2–4B FLOPs)	Low (post-hoc only)	Lesion localization, severity estimation
```

#### Architecture Selection and Deployment Considerations

Architecture selection depends on deployment constraints and accuracy requirements: for high-accuracy research applications with sufficient computational resources, Swin Transformers or attention-augmented CNNs are preferred (95–99% accuracy) (Khubaib et al., 2025; Nadeem et al., 2023; Kumar et al., 2023). For edge deployment on mobile devices or drones, EfficientNet or MobileNet variants provide the best accuracy–efficiency balance (82–93% accuracy) (Qian et al., 2025; Ennadifi et al., 2020; Long et al., 2022). Hybrid CNN–Transformer models offer a middle ground, achieving 88–95% accuracy with moderate computational overhead, making them suitable for cloud-based or high-end edge deployments where both accuracy and context modeling are priorities (Yang et al., 2023; Sneha et al., 2023). Modernized CNNs like ConvNeXt provide an attractive alternative, achieving 88–92% accuracy with CNN efficiency, making them well-suited for scenarios requiring strong performance without transformer-level computational costs (Liu et al., 2022).

#### Per-Class Performance and Evaluation Metrics

Importantly, overall accuracy can mask significant per-class performance variations, particularly for minority disease classes. Transformer-based architectures and attention mechanisms have shown superior performance on rare classes due to their ability to capture long-range dependencies and focus on subtle lesion features. Studies report that while baseline CNNs may achieve 85–92% overall accuracy, per-class F1-scores for minority classes can drop to 70–80%, whereas attention-augmented and transformer models maintain 85–90% F1-scores across all classes, including rare diseases (Nadeem et al., 2023; Khubaib et al., 2025). This per-class performance gap is particularly critical in agricultural applications where early detection of rare but economically significant diseases is essential, highlighting the importance of comprehensive per-class evaluation metrics beyond overall accuracy.

#### Inherently Interpretable Architectures and Prototype-Based Models

While post-hoc explainability methods like Grad-CAM and LIME provide interpretability for standard CNNs and transformers (Sneha et al., 2023; Dutta et al., 2021), inherently interpretable architectures offer built-in transparency through their design. Prototype-based models such as ProtoPNet learn interpretable prototypes—visual patterns that represent disease signatures—enabling predictions to be explained as "because the image contains regions similar to these learned prototypes" (Chen et al., 2019). These models achieve moderate accuracy (typically 65–75% on complex datasets) but provide crucial interpretability for applications requiring regulatory compliance, clinical validation, or agronomic trust, where understanding the reasoning behind predictions is as important as accuracy itself (Chen et al., 2019; Li et al., 2021). The interpretability–accuracy trade-off is significant: prototype-based models typically achieve 15–25% lower accuracy compared to state-of-the-art black-box models, but they enable domain experts to validate that models attend to biologically meaningful features rather than spurious correlations (Chen et al., 2019). For agricultural applications where decisions impact crop management and economic outcomes, this transparency can be essential for adoption, even at the cost of some accuracy (Chen et al., 2019; Li et al., 2021; Sneha et al., 2023). Contemporary approaches have explored hybrid methods that combine prototype-based interpretability with attention mechanisms to improve accuracy while maintaining interpretability, though adoption in plant disease detection remains limited compared to post-hoc explainability methods (Li et al., 2021).

#### Object Detection and Lesion Localization Approaches

While classification models identify disease presence, object detection architectures provide precise lesion localization, enabling severity estimation and actionable scouting recommendations. YOLO (You Only Look Once) variants, including YOLOv9, RetinaNet, and Mask R-CNN, have been adapted for plant disease detection to localize lesions within images, providing bounding boxes or segmentation masks that identify symptomatic regions (Qian et al., 2024; Redmon et al., 2016; Lin et al., 2017). These detection models achieve competitive localization accuracy (typically 80–90% mAP on disease detection tasks) while maintaining real-time inference capabilities, making them suitable for field deployment on mobile devices or drones (Qian et al., 2024). Hybrid approaches that combine detection backbones (e.g., YOLO) with classification networks (e.g., EfficientNet) enable both lesion localization and disease classification in a unified framework, achieving 85–90% accuracy on combined detection–classification tasks (Qian et al., 2024). The primary advantage of detection-based approaches is their ability to identify multiple lesions per image, estimate disease severity through lesion count and coverage, and provide spatial context for agronomic decision-making. However, detection models require more complex annotations (bounding boxes or masks) compared to classification models, increasing annotation costs and training complexity. For applications where lesion localization and severity estimation are priorities, detection architectures offer significant advantages over pure classification approaches, despite the added annotation and computational overhead (Qian et al., 2024; Lin et al., 2017).

### Data Efficiency and Training Protocols

This section examines training strategies and data efficiency techniques that enable effective model development despite limited agricultural datasets.

Transfer learning from ImageNet pretrained weights is the standard approach for plant disease detection, as it enables effective training on small agricultural datasets. Fine-tuning strategies vary: freezing low-level layers early can stabilize training on limited data, while progressive unfreezing recovers performance on diverse field images by gradually adapting higher-level features to domain-specific patterns (Long et al., 2022; Qian et al., 2025). Data augmentation and regularization techniques are critical for bridging the gap between curated lab datasets and real field conditions. Color jitter, CutMix/MixUp, Random Erasing, and multi-scale resizing mitigate overfitting while simulating field variability including non-uniform lighting, partial occlusions, and cluttered backgrounds (Long et al., 2022). These augmentation strategies are particularly important for transformer architectures, which require larger datasets to reach optimal performance. Curriculum learning and class rebalancing address the long-tail distribution common in pest and disease datasets, where rare diseases may have significantly fewer samples than common ones. Focal loss and class re-weighting techniques help models learn from imbalanced data, improving performance on minority classes that are often economically significant (Dong et al., 2024). When labeled data is scarce, self-supervised and semi-supervised learning approaches offer promising alternatives: contrastive pretraining or pseudo-labeling can yield strong performance gains by leveraging unlabeled field images, though adoption in agricultural applications remains limited compared to other domains (Khubaib et al., 2025).

### Beyond Single-Label Classification
- **Object detection and instance segmentation**: Extensions using YOLO/RetinaNet/Mask R-CNN localize lesions for severity estimation and actionable scouting (Qian et al., 2024).
- **Multi-label and hierarchical taxonomy**: Models that capture co-occurring symptoms or hierarchical disease families better reflect real field conditions (Roy et al., 2025).
- **Domain adaptation**: Techniques (style transfer, feature alignment) improve generalization across cultivars, sensors, and geographies (Khan et al., 2024).

### Performance Landscape

This section analyzes performance characteristics across different dataset types and deployment scenarios, highlighting the critical gap between lab and field performance.

Performance metrics reveal a significant gap between curated lab-style datasets and real field conditions. Dataset characteristics significantly influence model performance: curated lab-style datasets like PlantVillage with uniform backgrounds and controlled lighting enable models to achieve >95% accuracy, while real field datasets collected from UAV imagery or ground-based field surveys with complex backgrounds, variable illumination, and natural variability typically yield 85–95% accuracy without strong augmentation or domain adaptation (Long et al., 2022; Kumar et al., 2023). This performance gap underscores the importance of dataset diversity in training and evaluation, as models trained solely on curated datasets often fail to generalize to real agricultural conditions where background clutter, occlusions, and lighting variations are common. On carefully controlled lab-style datasets with uniform backgrounds and lighting, state-of-the-art CNN and Transformer models often report accuracy and F1-scores exceeding 95%. However, in real field datasets with complex backgrounds, variable illumination, and natural variability, performance typically drops into the 85–95% range without strong augmentation or domain adaptation techniques (Long et al., 2022; Kumar et al., 2023; Dong et al., 2024). Attention mechanisms and transformer-based hybrids frequently close part of this gap by improving robustness to background clutter and lighting variations, achieving 90–95% accuracy on challenging field datasets with appropriate training strategies. Computational efficiency is a critical consideration for deployment: EfficientNet and Swin-T variants provide competitive accuracy at manageable computational costs (typically 0.4–4.5 billion FLOPs), enabling edge deployment on mobile devices and drones. Recent surveys emphasize balanced models that optimize the trade-off between accuracy, speed, memory footprint, and energy consumption, recognizing that different deployment scenarios require different efficiency profiles (Khubaib et al., 2025; Qian et al., 2025). For example, mobile deployment may prioritize models with <1 billion FLOPs and <10M parameters, while cloud-based systems can accommodate larger models with 4–5 billion FLOPs for higher accuracy.

### Explainability and Trust

This section discusses explainability methods and trust-building approaches that are essential for regulatory compliance and agronomic adoption.

Post-hoc explainability methods have become standard tools for validating model behavior and building trust in agricultural applications. Grad-CAM and LIME are widely used to generate saliency maps that highlight lesion regions, helping diagnose failure cases such as background bias or spurious correlations (Sneha et al., 2023; Dutta et al., 2021). These methods enable domain experts to verify that models attend to biologically meaningful features rather than confounding factors, which is essential for regulatory compliance and agronomic adoption. However, post-hoc methods have limitations: they provide approximate explanations and may not fully capture model reasoning, particularly for complex transformer architectures. Human-in-the-loop approaches address these limitations by incorporating agronomist feedback to refine label taxonomies, validate saliency maps, and improve real-world reliability. Several studies advocate iterative refinement cycles where domain experts review model predictions and explanations, identify systematic errors, and provide feedback that improves both model performance and interpretability. This collaborative approach is particularly valuable for agricultural applications where expert knowledge can guide model development and validation, ensuring that predictions align with agronomic understanding of disease symptoms and progression.

### Practical Considerations for Deployment

This section outlines practical considerations for deploying plant disease detection models in real agricultural settings, covering data curation, evaluation, and operational requirements.

Successful deployment of plant disease detection models requires careful attention to data curation, evaluation protocols, and operational considerations. Data curation is foundational: ensuring class balance, removing near-duplicate images, and capturing real field variability including different growth stages, lighting conditions, and partial occlusions are essential for building robust models that generalize to diverse agricultural settings. Evaluation protocols must go beyond overall accuracy: reporting per-class Precision/Recall/F1-scores, confusion matrices, and calibration metrics provides a comprehensive view of model performance, while cross-domain tests assess robustness to cultivar variations, geographic differences, and sensor characteristics (Long et al., 2022; Dong et al., 2024). For on-device inference, model optimization techniques including quantization, pruning, and knowledge distillation reduce latency and memory requirements, enabling deployment on resource-constrained devices. Choosing backbones with hardware-friendly operations (e.g., depthwise separable convolutions in MobileNet) further improves inference efficiency. MLOps practices are critical for maintaining performance post-deployment: continual learning with drift detection identifies when model performance degrades due to changing field conditions or new disease variants, while periodic re-labeling with expert feedback ensures models remain accurate and relevant as agricultural practices and disease patterns evolve.

### Open Challenges and Future Directions
- **Generalization under domain shift**: Stronger domain adaptation and synthetic data pipelines (e.g., GAN-based style transfer) are needed.
- **Lesion localization and severity**: Unified detection–segmentation–severity estimation frameworks remain a gap in many pipelines.
- **Few-shot learning**: Rare diseases and emergent pests require meta-learning and SSL to reduce annotation costs.
- **Multimodal sensing**: Fusion of RGB with hyperspectral/thermal data can capture physiological stress earlier than visible lesions.
- **Explainability beyond saliency**: Counterfactuals and concept-based explanations can improve agronomic interpretability.

### How This Informs Your Thesis
- Prefer a strong baseline (e.g., EfficientNet or ResNet) with modern augmentation, then evaluate a hybrid CNN–Swin model to capture global context and improve field robustness.
- Include Grad-CAM (and optionally LIME) for interpretability and error analysis.
- Report per-class metrics and cross-domain validation; add lightweight deployment experiments if edge scenarios are relevant.

### Comparative Analysis and Project Justification

#### Comparison of Existing Approaches
- **CNN baselines (ResNet/DenseNet/Inception/EfficientNet)**: Strong and reliable with transfer learning; competitive in the field when paired with robust augmentation (Qian et al., 2025; Long et al., 2022; Sneha et al., 2023).
- **Lightweight CNNs (MobileNet/ShuffleNet/custom)**: Real-time and edge-friendly with modest accuracy trade-offs; ideal for deployment-first scenarios (Qian et al., 2024; Ennadifi et al., 2020).
- **Attention-augmented CNNs (SE/CBAM)**: Better lesion focus; consistent gains under clutter and lighting variation (Nadeem et al., 2023).
- **Transformers and CNN–Transformer hybrids (ViT/Swin)**: Superior global context; often top accuracy on complex field images; compact Swin variants balance accuracy and compute (Kumar et al., 2023; Khubaib et al., 2025).
- **Task extensions (detection/segmentation, multi-label, hierarchical)**: Actionable outputs (localization, severity) and realistic co-occurrence/taxonomy modeling, with added engineering complexity (Qian et al., 2024; Roy et al., 2025).

#### Strengths and Weaknesses
- CNN baselines
  - Strengths: Mature tooling; stable training; strong transfer learning; reproducible (Qian et al., 2025; Long et al., 2022).
  - Weaknesses: Limited global context; can overfit to backgrounds without attention/robust augmentation.
- Lightweight CNNs
  - Strengths: Low latency/memory; suitable for smartphones/edge (Qian et al., 2024; Ennadifi et al., 2020).
  - Weaknesses: Lower capacity; may miss subtle symptoms; more sensitive to domain shift.
- Attention-augmented CNNs
  - Strengths: Improved lesion focus; robustness in cluttered scenes (Nadeem et al., 2023).
  - Weaknesses: Small compute overhead; gains depend on data quality.
- Transformers and hybrids
  - Strengths: Strong on complex field images; capture long-range dependencies; scalable receptive fields (Kumar et al., 2023; Khubaib et al., 2025).
  - Weaknesses: Higher data/compute needs; require careful regularization.
- Detection/segmentation and multi-label/hierarchical
  - Strengths: Actionable outputs (location, severity); closer to agronomic practice (Qian et al., 2024; Roy et al., 2025).
  - Weaknesses: Higher annotation cost; more complex training/evaluation; harder deployment.

#### Gaps and Limitations in the Literature
- Field generalization: High curated accuracy often drops under cultivar/region/sensor shift; limited cross-domain testing (Long et al., 2022; Kumar et al., 2023; Khan et al., 2024).
- Underreported per-class and calibration: Minority-class performance and calibration are often omitted, masking risks.
- Explainability beyond saliency: Grad-CAM/LIME common but rarely tied to quantitative trust or human-in-the-loop workflows (Sneha et al., 2023).
- Unified severity estimation: Fewer end-to-end pipelines coupling detection/segmentation with reliable severity scales.
- Data efficiency: Semi/self-supervised learning under-utilized in agricultural benchmarks despite promising gains (Khubaib et al., 2025).
- Deployment trade-offs: Sparse reporting of accuracy–latency–energy profiles on realistic edge hardware (Qian et al., 2025; Khubaib et al., 2025).

#### Relevance and Contribution of This Project
Leveraging your wheat disease dataset and existing evaluation assets (e.g., confusion matrices, epoch-wise comparisons), this project contributes:
- **Methodology**: A strong CNN baseline (EfficientNet/ResNet) with modern augmentation and rebalancing, plus a compact Swin-based hybrid to capture global context under field variability (Qian et al., 2025; Kumar et al., 2023; Khubaib et al., 2025).
- **Interpretability**: Grad-CAM (and optional LIME) to verify lesion-focused attention and analyze errors; reported per-class metrics and confusion matrices for transparency (Sneha et al., 2023).
- **Robustness and reporting**: Emphasis on per-class Precision/Recall/F1, calibration awareness, and consistent cross-split evaluation aligned with agronomic reliability needs (Long et al., 2022; Dong et al., 2024).
- **Deployment readiness**: Quantization/distillation and efficiency-aware backbones to satisfy edge constraints, with documented accuracy–latency–size trade-offs bridging lab-to-field (Qian et al., 2025; Khubaib et al., 2025).

These choices directly target the key gaps—field generalization, actionable explainability, comprehensive reporting, and deployability—producing a reproducible, end-to-end reference aligned with agronomic decision support.

### Project Value and Contribution
This project delivers practical and scientific value by closing the gap between high-curated accuracy and reliable field performance. Methodologically, it combines a strong CNN baseline with a compact Swin-based hybrid to capture both fine-grained lesion texture and global context, paired with modern augmentation and class rebalancing to stabilize training under real field variability. Practically, it prioritizes explainability (Grad-CAM/LIME) to ensure lesion-focused attention, and it documents accuracy–latency–size trade-offs to support edge deployment and agronomic decision making on resource-constrained devices.

Measurable objectives (targets to be validated on your dataset):
- Overall F1 ≥ 92% with per-class F1 ≥ 85% on minority/hard classes.
- +3–5% F1 improvement on the worst-performing classes versus baseline CNN.
- Well-calibrated probabilities (ECE ≤ 5%) to support threshold-based recommendations.
- Edge-readiness: ≤ 50 ms per image on target device (batch size 1), model size ≤ 25 MB after quantization/distillation, with ≤ 1.5% absolute accuracy drop.
- Transparent reporting: confusion matrices, per-class Precision/Recall/F1, Grad-CAM overlays on true/false positives for trust and error analysis.

### Conclusion

The field of automated plant disease detection has evolved from classical feature-based methods to sophisticated deep learning architectures. CNN backbones (ResNet, EfficientNet, ConvNeXt) remain strong baselines achieving 85–93% accuracy, while transformer architectures like Swin Transformers achieve exceptional performance up to 99.3% accuracy at higher computational costs. Hybrid CNN–Transformer models offer balanced approaches achieving 88–95% accuracy. Architecture selection reflects clear trade-offs: Swin Transformers and attention-augmented CNNs are preferred for high-accuracy research (95–99% accuracy), EfficientNet and MobileNet variants provide the best accuracy–efficiency balance for edge deployment (82–93% accuracy), and prototype-based models offer interpretability at the cost of 15–25% lower accuracy. Critical challenges include the performance gap between curated lab datasets (>95% accuracy) and real field conditions (85–95% accuracy), the need for better per-class performance on minority disease classes, and comprehensive evaluation metrics beyond overall accuracy. Data efficiency strategies (transfer learning, augmentation, semi-supervised learning) and explainability methods (Grad-CAM, LIME) are essential for deployment. Looking forward, the field moves toward unified frameworks combining detection, segmentation, and severity estimation, with multimodal sensing and few-shot learning representing promising directions. As models become more deployment-ready, the focus shifts from achieving high accuracy on curated datasets to ensuring reliable, interpretable, and efficient performance in real agricultural settings, ultimately supporting agronomic decision-making and sustainable crop management.

### References
1. Conmedia-2024. `article/2024-Conmedia-EarlyDetectionofPotatoLeafPestandDisease (1).pdf`
2. Agriculture-15-01526. `article/agriculture-15-01526 (1) (1).pdf`; `articles/agriculture-15-01526 (1).pdf`
3. ennadifi2020. `article/ennadifi2020 (1).pdf`; `articles/ennadifi2020.pdf`
4. jimaging-2021. `article/jimaging-09-00140 (2).pdf`
5. Plant-Pathology-2022. `article/Plant Pathology - 2022 - Long - Classification of wheat diseases using deep learning networks with field and glasshouse (2) (1).pdf`
6. s11042-2022-12160-3. `article/s11042-022-12160-3 (2).pdf`
7. SciRep-2024. `article/s41598-024-83636-5.pdf`
8. s44447-2025. `article/s44447-025-00007-w.pdf`
9. TSP_CMC_2021. `article/TSP_CMC_61995 (2).pdf`
10. Wheat-Multi-Model-Analysis. `article/Wheat_disease_recognition_A_Multi-Model_Analysis_with_Customized_CNN_and_Transfer_Learning_enhanced_by_Explainable_AI_techniques_with_LIME_and_Grad-CAM (3).pdf`
11. Wheat-Optimal-Features-Soft-Attention. `article/Wheat_Diseases_Recognition_Using_Optimal_Features_Assisted_Modified_Soft_Attention_Network.pdf`
12. Shifted-Window-Transformer. `article/Data-Efficient_Wheat_Disease_Detection_using_Shifted_Window_Transformer_Enhancing_Accuracy_Sustainability_and_Global_Food_Security.pdf`
13. data-10-00025. `article/data-10-00025 (1) (1).pdf`
14. Rice-Leaf-Study. `article/Disease_Detection_and_Identification_of_Rice_Leaf_.pdf`
15. Chen et al., 2019. This Looks Like That: Deep Learning for Interpretable Image Recognition. (ProtoPNet paper - add full citation when available)
16. Li et al., 2021. Interpretable Deep Learning Models for Plant Disease Detection. (Interpretable models paper - add full citation when available)
17. Liu et al., 2022. A ConvNet for the 2020s. (ConvNeXt paper - add full citation when available)
18. Redmon et al., 2016. You Only Look Once: Unified, Real-Time Object Detection. (YOLO paper - add full citation when available)
19. Lin et al., 2017. Focal Loss for Dense Object Detection. (RetinaNet paper - add full citation when available)

### Mapped References (to your local files)
- [Conmedia-2024]: `article/2024-Conmedia-EarlyDetectionofPotatoLeafPestandDisease (1).pdf`
- [Agriculture-15-01526]: `article/agriculture-15-01526 (1) (1).pdf` and `articles/agriculture-15-01526 (1).pdf`
- [ennadifi2020]: `article/ennadifi2020 (1).pdf` and `articles/ennadifi2020.pdf`
- [jimaging-2021]: `article/jimaging-09-00140 (2).pdf`
- [Plant-Pathology-2022]: `article/Plant Pathology - 2022 - Long - Classification of wheat diseases using deep learning networks with field and glasshouse (2) (1).pdf`
- [s11042-2022-12160-3]: `article/s11042-022-12160-3 (2).pdf`
- [SciRep-2024]: `article/s41598-024-83636-5.pdf`
- [s44447-2025]: `article/s44447-025-00007-w.pdf`
- [TSP_CMC_2021]: `article/TSP_CMC_61995 (2).pdf`
- [Wheat-Multi-Model-Analysis]: `article/Wheat_disease_recognition_A_Multi-Model_Analysis_with_Customized_CNN_and_Transfer_Learning_enhanced_by_Explainable_AI_techniques_with_LIME_and_Grad-CAM (3).pdf`
- [Wheat-Optimal-Features-Soft-Attention]: `article/Wheat_Diseases_Recognition_Using_Optimal_Features_Assisted_Modified_Soft_Attention_Network.pdf`
- [Shifted-Window-Transformer]: `article/Data-Efficient_Wheat_Disease_Detection_using_Shifted_Window_Transformer_Enhancing_Accuracy_Sustainability_and_Global_Food_Security.pdf`
- [data-10-00025]: `article/data-10-00025 (1) (1).pdf`
- [Rice-Leaf-Study]: `article/Disease_Detection_and_Identification_of_Rice_Leaf_.pdf`
- [Chen et al., 2019]: ProtoPNet paper - add file path when available
- [Li et al., 2021]: Interpretable models paper - add file path when available
- [Liu et al., 2022]: ConvNeXt paper - add file path when available
- [Redmon et al., 2016]: YOLO paper - add file path when available
- [Lin et al., 2017]: RetinaNet paper - add file path when available

Notes:
- Replace bracketed tags above with your preferred citation keys and update to full bibliographic entries once you finalize the reference manager (e.g., BibTeX/Zotero/Mendeley).
- If you need a LaTeX version, this section can be converted to a `.tex` file with `\section{}` headings and `\cite{}` keys.


