#!/usr/bin/env python3
"""
Create Confusion Matrix from JSON Labels
=======================================
This script reads true/predicted labels from a JSON file and generates a
confusion matrix heatmap. It can also augment an existing report JSON by
inserting the computed matrix under the 'confusion_matrix' key.

Accepted labels JSON schemas:
1) Array fields:
   {
     "y_true": ["classA", "classB", ...],
     "y_pred": ["classA", "classC", ...],
     "class_names": ["classA", "classB", ...]   # optional
   }

2) Pairs list:
   {
     "pairs": [
       {"true": "classA", "pred": "classB"},
       {"true": "classB", "pred": "classB"}
     ],
     "class_names": ["classA", "classB", ...]   # optional
   }

Usage examples:
  python create_confusion_from_json_labels.py \
    --labels-json parsed_outputs/epoch10/convnext/labels.json \
    --out-img confusion_matrices/epoch10/convnext/confusion_matrix.png

  python create_confusion_from_json_labels.py \
    --labels-json parsed_outputs/epoch10/convnext/labels.json \
    --report-json parsed_outputs/epoch10/convnext/report_with_optional_cm.json \
    --write-report parsed_outputs/epoch10/convnext/report_filled.json \
    --out-img confusion_matrices/epoch10/convnext/confusion_matrix.png

Notes:
- If class names are not provided, they are inferred from all unique labels
  appearing in y_true and y_pred (sorted by first appearance order).
- Set --classes to override class names explicitly (comma-separated).
- Use --normalize to plot normalized rates instead of raw counts.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_CLASS_NAMES = [
    'aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf',
    'spetoria', 'tan_spot', 'yellow_rust'
]


def read_labels(labels_path: Path) -> Tuple[List[str], List[str], List[str]]:
    data = json.loads(labels_path.read_text(encoding='utf-8'))

    y_true: List[str] = []
    y_pred: List[str] = []

    if isinstance(data, dict):
        if 'y_true' in data and 'y_pred' in data:
            y_true = list(map(str, data['y_true']))
            y_pred = list(map(str, data['y_pred']))
        elif 'pairs' in data and isinstance(data['pairs'], list):
            for p in data['pairs']:
                y_true.append(str(p.get('true')))
                y_pred.append(str(p.get('pred')))
        else:
            raise ValueError("Labels JSON must contain 'y_true'/'y_pred' or 'pairs'.")
    else:
        raise ValueError('Labels JSON root must be an object/dict.')

    if len(y_true) != len(y_pred):
        raise ValueError(f"Mismatched lengths: y_true={len(y_true)} vs y_pred={len(y_pred)}")

    class_names: List[str]
    if 'class_names' in data and isinstance(data['class_names'], list) and data['class_names']:
        class_names = [str(c) for c in data['class_names']]
    else:
        # Infer class names by first appearance in y_true then y_pred
        seen: Dict[str, bool] = {}
        order: List[str] = []
        for lbl in y_true + y_pred:
            if lbl not in seen:
                seen[lbl] = True
                order.append(lbl)
        class_names = order if order else DEFAULT_CLASS_NAMES

    return y_true, y_pred, class_names


def build_confusion_matrix(y_true: List[str], y_pred: List[str], class_names: List[str]) -> np.ndarray:
    index_by_class = {c: i for i, c in enumerate(class_names)}
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in index_by_class or p not in index_by_class:
            # Extend matrix if a label not in class_names appears
            # Add new class at the end
            if t not in index_by_class:
                index_by_class[t] = len(index_by_class)
                class_names.append(t)
                # resize matrix
                cm = _resize_cm(cm, len(index_by_class))
            if p not in index_by_class:
                index_by_class[p] = len(index_by_class)
                class_names.append(p)
                cm = _resize_cm(cm, len(index_by_class))
        i = index_by_class[t]
        j = index_by_class[p]
        cm[i, j] += 1
    return cm


def _resize_cm(cm: np.ndarray, new_size: int) -> np.ndarray:
    n_old = cm.shape[0]
    if new_size <= n_old:
        return cm
    cm_exp = np.zeros((new_size, new_size), dtype=int)
    cm_exp[:n_old, :n_old] = cm
    return cm_exp


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: Path, normalize: bool = False) -> None:
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm, row_sums, where=row_sums != 0)
        fmt = '.2f'
        cmap = 'Blues'
    else:
        cm_display = cm
        fmt = 'd'
        cmap = 'Blues'

    plt.figure(figsize=(max(8, 0.6 * len(class_names)), max(6, 0.5 * len(class_names))))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()


def augment_report_json(report_path: Path, out_path: Path, cm: np.ndarray, class_names: List[str]) -> None:
    try:
        data = json.loads(report_path.read_text(encoding='utf-8'))
    except Exception:
        data = {}
    data['confusion_matrix'] = cm.tolist()
    # If per_class already exists, keep it; ensure class ordering if helpful
    if 'per_class' in data and isinstance(data['per_class'], dict):
        # Optionally reorder per_class to match class_names
        ordered = {k: data['per_class'].get(k, {}) for k in class_names}
        # Include any extras that were not in class_names
        for k, v in data['per_class'].items():
            if k not in ordered:
                ordered[k] = v
        data['per_class'] = ordered
    # Write out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-json', required=True, help='JSON file with y_true/y_pred or pairs list')
    parser.add_argument('--report-json', default=None, help='Existing report JSON to augment with confusion_matrix')
    parser.add_argument('--write-report', default=None, help='Path to write the augmented report JSON')
    parser.add_argument('--out-img', required=True, help='Path to save the confusion matrix image (PNG)')
    parser.add_argument('--title', default='Confusion Matrix', help='Title for the plot')
    parser.add_argument('--classes', default=None, help='Comma-separated override for class names')
    parser.add_argument('--normalize', action='store_true', help='Plot normalized rates instead of raw counts')
    args = parser.parse_args()

    labels_path = Path(args.labels_json)
    y_true, y_pred, class_names = read_labels(labels_path)

    if args.classes:
        class_names = [c.strip() for c in args.classes.split(',') if c.strip()]

    cm = build_confusion_matrix(y_true, y_pred, class_names)

    out_img = Path(args.out_img)
    plot_confusion_matrix(cm, class_names, args.title, out_img, normalize=args.normalize)
    print(f"[IMG] {out_img}")

    if args.report_json and args.write_report:
        report_in = Path(args.report_json)
        report_out = Path(args.write_report)
        augment_report_json(report_in, report_out, cm, class_names)
        print(f"[REPORT] {report_out}")


if __name__ == '__main__':
    main()
