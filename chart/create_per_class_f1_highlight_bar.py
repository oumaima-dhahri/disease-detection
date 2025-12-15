"""
Per-class F1 grouped bar chart with highlighted easy/hard classes.
Generates `chart/per_class_f1_highlight_bar.png`.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Class order and per-class F1 (0–1 range) for each model
classes = [
    "aphid",
    "army_worm",
    "black_rust",
    "brown_rust",
    "common_rust",
    "fusarium_head_blight",
    "healthy",
    "leaf_blight",
    "powdery_mildew_leaf",
    "spetoria",
    "tan_spot",
    "yellow_rust",
]

convnext = [0.9195, 0.9882, 0.9130, 0.9556, 0.9811, 0.9859, 0.9589, 0.6889, 0.9074, 0.9647, 0.6389, 1.0000]
hybrid_cnn_vit = [0.9157, 0.9767, 0.8511, 0.9451, 0.9720, 0.9722, 0.9589, 0.7294, 0.9273, 0.9535, 0.6111, 1.0000]
hybrid_v2 = [0.9286, 0.9655, 0.8478, 0.9425, 0.9286, 0.9444, 0.9379, 0.6420, 0.9245, 0.9524, 0.6829, 1.0000]
yolo_effnet = [0.8941, 0.9655, 0.8911, 0.9655, 0.9231, 0.9444, 0.9452, 0.6341, 0.9143, 0.9639, 0.6250, 1.0000]

model_data = [convnext, hybrid_cnn_vit, hybrid_v2, yolo_effnet]
model_names = ["ConvNeXt", "Hybrid CNN-ViT", "Hybrid V2", "YOLOv9+EffNet"]
# Clean legend colors (user-specified palette)
convnext_color = "#003f5c"         # deep navy blue – authoritative
hybrid_cnn_vit_color = "#58508d"    # muted purple – distinct, less dominant
hybrid_v2_color = "#bc5090"         # dusty rose – warm, avoids red-green confusion
yolo_effnet_color = "#ffa600"       # amber/orange – vibrant but professional
colors_default = [
    convnext_color,
    hybrid_cnn_vit_color,
    hybrid_v2_color,
    yolo_effnet_color,
]


def main() -> None:
    x = np.arange(len(classes))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (model, name, base_color) in enumerate(zip(model_data, model_names, colors_default)):
        offsets = x + (i - 1.5) * width
        edge_kwargs = {"linewidth": 1.2, "edgecolor": "#1f3d6d"} if name == "ConvNeXt" else {"linewidth": 0.6, "edgecolor": "white"}
        alpha = 0.95 if name == "ConvNeXt" else 0.9
        ax.bar(offsets, model, width, label=name, color=base_color, alpha=alpha, **edge_kwargs)

    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    # No legend per user request
    ax.set_title("Per-Class F1-Score Comparison Across Four Deep Learning Models")

    # Light horizontal gridlines
    ax.grid(axis="y", linestyle="-", linewidth=0.5, color="#d8d8d8", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.0, 1, 0.92])

    out_path = Path(__file__).with_name("per_class_f1_highlight_bar.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

