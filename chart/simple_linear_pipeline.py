#!/usr/bin/env python3
"""
Simple Linear Pipeline Diagram
Creates a clean horizontal diagram:
Data → Preprocessing → Model → Training → Evaluation → Interpretability
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch


def draw_box(ax, x, y, w, h, label, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=11, fontweight='bold')


def create_simple_linear_pipeline(output_png='simple_pipeline.png', output_pdf='simple_pipeline.pdf'):
    fig, ax = plt.subplots(1, 1, figsize=(14, 3.5))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Colors
    colors = {
        'data': '#E3F2FD',           # Light blue
        'pre': '#E0F7FA',            # Cyan
        'model': '#FFF3E0',          # Light orange
        'train': '#E8F5E9',          # Light green
        'eval': '#FCE4EC',           # Light pink
        'interp': '#F3E5F5'          # Light purple
    }

    # Box geometry
    w = 4.2
    h = 2.0
    y = 1.5
    xs = [1, 6.2, 11.4, 16.6, 21.8, 27.0]
    labels = [
        'DATA',
        'PREPROCESSING',
        'MODEL',
        'TRAINING',
        'EVALUATION',
        'INTERPRETABILITY'
    ]
    color_keys = ['data', 'pre', 'model', 'train', 'eval', 'interp']

    # Draw boxes
    for x, label, ck in zip(xs, labels, color_keys):
        draw_box(ax, x, y, w, h, label, colors[ck])

    # Draw arrows
    for i in range(len(xs) - 1):
        start = (xs[i] + w, y + h/2)
        end = (xs[i+1], y + h/2)
        arrow = ConnectionPatch(start, end, "data", "data",
                                arrowstyle="->", mutation_scale=18,
                                shrinkA=4, shrinkB=4, fc="black", lw=2)
        ax.add_patch(arrow)

    # Title
    ax.text(15, 4.5, 'Wheat Disease Detection Pipeline', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_png}\nSaved: {output_pdf}")


if __name__ == '__main__':
    create_simple_linear_pipeline()





