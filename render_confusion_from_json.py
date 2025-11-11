#!/usr/bin/env python3
"""
Render Confusion Matrix from JSON
================================
Reads JSON file(s) and renders confusion matrix images if 'confusion_matrix' is
provided. Also exports per-class metrics to CSV if present.

Usage (single file):
  python render_confusion_from_json.py --json parsed_outputs/epoch10/convnext/report_filled.json \
                                       --out_dir cm_outputs/epoch10/convnext \
                                       --title "ConvNeXt - Confusion Matrix (epoch10)"

Usage (batch over a directory, recursive):
  python render_confusion_from_json.py --json-dir parsed_outputs --out_dir cm_outputs
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_CLASS_NAMES = [
    'aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf',
    'spetoria', 'tan_spot', 'yellow_rust'
]

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, save_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()

def write_metrics_csv(per_class: dict, out_csv: Path) -> None:
    import csv
    headers = ['class', 'precision', 'recall', 'f1_score', 'support']
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for cls in per_class:
            m = per_class[cls] or {}
            writer.writerow([
                cls,
                m.get('precision', ''),
                m.get('recall', ''),
                m.get('f1_score', ''),
                m.get('support', ''),
            ])

def resolve_class_names(data: dict, override: Optional[str]) -> List[str]:
    if override:
        return [c.strip() for c in override.split(',')]
    # Prefer per_class keys order if provided
    if isinstance(data.get('per_class'), dict) and data['per_class']:
        return list(data['per_class'].keys())
    return DEFAULT_CLASS_NAMES

def process_json_file(json_path: Path, out_root: Path, title_override: Optional[str], class_names_override: Optional[str]) -> None:
    try:
        data = json.loads(json_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"[ERROR] Failed to read {json_path}: {e}")
        return

    # Mirror directory structure under out_root
    # e.g., parsed_outputs/epoch10/convnext/report.json -> cm_outputs/epoch10/convnext/
    rel_parent = json_path.parent
    out_dir = out_root / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write per-class CSV if available
    if 'per_class' in data and isinstance(data['per_class'], dict) and data['per_class']:
        write_metrics_csv(data['per_class'], out_dir / 'per_class_metrics.csv')
        print(f"[CSV] {out_dir / 'per_class_metrics.csv'}")

    # Plot confusion matrix if provided
    cm = data.get('confusion_matrix')
    if cm:
        cm_np = np.array(cm, dtype=int)
        title = title_override or f"{data.get('model', 'Model')} - Confusion Matrix ({data.get('epoch', '')})"
        class_names = resolve_class_names(data, class_names_override)
        plot_confusion_matrix(cm_np, class_names, title, out_dir / 'confusion_matrix.png')
        print(f"[IMG] {out_dir / 'confusion_matrix.png'}")
    else:
        print(f"[SKIP] No confusion_matrix in {json_path}")

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--json', help='Single JSON file to render')
    group.add_argument('--json-dir', help='Directory to scan recursively for JSON files')
    parser.add_argument('--out_dir', required=True, help='Output directory for image/CSV')
    parser.add_argument('--title', default=None, help='Custom title for plots (single file mode only or used for all if provided)')
    parser.add_argument('--class-names', default=None, help='Comma-separated class names override')
    args = parser.parse_args()

    out_root = Path(args.out_dir)

    if args.json:
        process_json_file(Path(args.json), out_root, args.title, args.class_names)
        return

    # Batch mode: iterate all JSON files under json-dir
    json_dir = Path(args.json_dir).resolve()
    for root, _, files in os.walk(json_dir):
        for fn in files:
            if not fn.lower().endswith('.json'):
                continue
            json_path = Path(root) / fn
            process_json_file(json_path, out_root, args.title, args.class_names)

if __name__ == '__main__':
    main()
