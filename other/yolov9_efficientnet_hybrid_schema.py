"""
YOLOv9 + EfficientNet-B3 Hybrid Architecture Schematic

This script generates a high-resolution diagram illustrating a pipeline where
YOLOv9 performs lesion localization and confidence scoring, then crops ROIs
that are classified by EfficientNet-B3. The figure is saved to
"yolov9_efficientnet_hybrid_schema.png" in the project root.

Requirements: matplotlib
Run: python yolov9_efficientnet_hybrid_schema.py
"""

import os
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, Rectangle


def add_box(ax, xy, width, height, text, fc="#f5f5f5", ec="#333333", fontsize=11):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=6",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
    )
    return box


def add_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle=ArrowStyle("Simple", head_length=8, head_width=6, tail_width=1.6),
            color="#444444",
            lw=1.2,
        ),
    )


def main():
    fig_w, fig_h = 13, 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_axis_off()
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 70)

    # Outer rounded container
    outer = FancyBboxPatch((2, 4), 116, 64, boxstyle="round,pad=0.02,rounding_size=14",
                           linewidth=1.4, edgecolor="#9aa0a6", facecolor="#ffffff")
    ax.add_patch(outer)

    # Title banner
    banner = FancyBboxPatch((36, 62), 48, 6, boxstyle="round,pad=0.02,rounding_size=8",
                            linewidth=0, facecolor="#d8efd3")
    ax.add_patch(banner)
    ax.text(60, 65, "Proposed YOLOv9 + EfficientNet-B3 Hybrid Model", ha="center", va="center",
            fontsize=13, color="#2f4f2f", fontweight="bold")

    # Row 1: Input → YOLOv9 backbone/neck/head
    input_box = add_box(ax, (10, 44), 16, 8, "Input Image\n(Leaf)", fc="#ffe6e6")
    yolo_backbone = add_box(ax, (32, 44), 18, 8, "YOLOv9 Backbone\n(CSP/GELAN)", fc="#e6f0ff")
    yolo_neck = add_box(ax, (54, 44), 16, 8, "FPN / PAN Neck\nMulti-scale", fc="#e8f7ff")
    yolo_head = add_box(ax, (74, 44), 18, 8, "Decoupled Head\nBoxes + Scores", fc="#ecfff1")

    add_arrow(ax, (26, 48), (32, 48))
    add_arrow(ax, (50, 48), (54, 48))
    add_arrow(ax, (70, 48), (74, 48))

    # Row 2: ROI cropping branch from detections → EfficientNet-B3 classifier
    nms = add_box(ax, (30, 34), 22, 8, "NMS + Confidence\nThresholding", fc="#fff3f3")
    roi_sel = add_box(ax, (32, 26), 22, 8, "ROI Selection /\nCrop from Boxes", fc="#fff6e6")
    eff_b3 = add_box(ax, (58, 26), 22, 8, "EfficientNet-B3\nClassifier (MBConv, SE)\n(Batched ROIs)", fc="#f4e8ff")
    decision = add_box(ax, (86, 26), 22, 8, "Decision + Confidence\nPer-Box Class Labels", fc="#f0fff2")

    # Global branch: whole-image classification in parallel
    global_branch = add_box(ax, (96, 44), 18, 8, "Global Leaf\nClassification", fc="#e9f7ef")
    # Route global branch from input image directly
    add_arrow(ax, (26, 48), (96, 48))

    # Drop arrows from YOLO head (bottom) to ROI selection (top)
    # Head → NMS/thresholding
    add_arrow(ax, (83, 44), (52, 38))
    add_arrow(ax, (74, 44), (41, 38))
    # NMS → ROI selection
    add_arrow(ax, (41, 34), (43, 30))

    add_arrow(ax, (54, 30), (58, 30))
    add_arrow(ax, (80, 30), (86, 30))

    # Side panel: metadata
    meta = (
        "YOLOv9: PGI, GELAN, decoupled head — multi-scale lesion localization\n"
        "EfficientNet-B3: compound scaling (depth/width/res), MBConv + SE\n"
        "Pipeline: detect lesions → crop ROIs → classify each ROI → per-box labels"
    )
    ax.text(8, 12, meta, fontsize=10.2, color="#333333")

    # Caption / references
    caption = (
        "Schematic of the YOLOv9 + EfficientNet-B3 hybrid pipeline: YOLOv9 localizes"
        " symptoms and outputs confidence-scored boxes → NMS/thresholding → ROI crops"
        " classified by EfficientNet-B3 (batched). In parallel, a global leaf"
        " classification head predicts an overall image label. References: Ultralytics"
        " (YOLOv9 docs), Tan & Le (ICML 2019, EfficientNet)."
    )
    ax.text(8, 6, caption, fontsize=9.8, color="#555555")

    out_path = os.path.abspath("yolov9_efficientnet_hybrid_schema.png")
    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


