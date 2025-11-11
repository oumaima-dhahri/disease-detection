#!/usr/bin/env python3
"""
Parse Classification Report and (Optionally) Plot Confusion Matrix
=================================================================
Given a training/evaluation log file, this utility will:
- Extract the scikit-learn style classification report table and save it as JSON
- Try to find a printed numeric confusion matrix block and, if present, render and save a confusion matrix heatmap

Usage:
  python parse_report_and_plot_cm.py --log "epoch10/output trainig/train convnext.txt" \
                                     --out_dir parsed_outputs/epoch10/convnext \
                                     --model_name "ConvNeXt" --epoch_label "epoch10"

Outputs:
  out_dir/
    classification_report.json
    confusion_matrix.png  (only if matrix values found)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Default class names (override via --class_names if needed)
DEFAULT_CLASS_NAMES = [
    'aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf',
    'spetoria', 'tan_spot', 'yellow_rust'
]


def parse_classification_report(text: str) -> Optional[Dict]:
    """Parse a scikit-learn classification report table into a structured dict.
    Returns None if not found.
    """
    lines = text.splitlines()
    parsed: Dict[str, Dict[str, float]] = {}

    started = False
    any_row = False
    for line in lines:
        raw = line.strip()
        if not raw:
            # allow blank lines until we've captured at least one row
            if started and any_row:
                break
            else:
                continue
        low = raw.lower()
        if not started and ('f1' in low and 'support' in low and 'recall' in low and 'precision' in low):
            # header line
            started = True
            continue
        if not started:
            continue
        # Try to parse a data row
        parts = re.split(r"\s+", raw)
        # Special case: accuracy row
        if parts and parts[0].lower() == 'accuracy':
            try:
                value = float(parts[-2])
                total = int(parts[-1])
                parsed['accuracy'] = {'accuracy': value, 'support': total}
                any_row = True
            except Exception:
                pass
            continue
        if len(parts) < 5:
            # Not enough tokens to be a data row
            continue
        try:
            support = int(parts[-1])
            f1 = float(parts[-2])
            recall = float(parts[-3])
            precision = float(parts[-4])
            name = " ".join(parts[:-4])
            if not name:
                continue
            parsed[name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support,
            }
            any_row = True
        except Exception:
            if parsed:
                break
            else:
                continue

    return parsed if parsed else None


def parse_confusion_matrix_block(text: str) -> Optional[np.ndarray]:
    """Extract a printed numeric confusion matrix from a log (as nested lists)."""
    m = re.search(r"Confusion matrix values:\s*\n(\[.+?\])", text, flags=re.S)
    block = m.group(1) if m else None
    if not block:
        # fallback: any nested list block
        m2 = re.search(r"(\[\s*\[.+?\]\s*\])", text, flags=re.S)
        block = m2.group(1) if m2 else None
    if not block:
        return None
    rows = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not (line.startswith('[') and line.endswith(']')):
            continue
        inner = line[1:-1].strip()
        if not inner:
            continue
        parts = re.split(r"[\s,]+", inner)
        try:
            nums = [int(p) for p in parts if p]
        except Exception:
            return None
        rows.append(nums)
    if not rows:
        return None
    try:
        mat = np.array(rows, dtype=int)
    except Exception:
        return None
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return None
    return mat


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True, help='Path to training/evaluation log file')
    parser.add_argument('--out_dir', required=True, help='Directory to write JSON and image outputs')
    parser.add_argument('--model_name', default='Model', help='Pretty model name for titles')
    parser.add_argument('--epoch_label', default='epoch', help='Label for epoch (e.g., epoch10)')
    parser.add_argument('--class_names', default=None, help='Comma-separated class names to use on axes')
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(errors='ignore')

    # Parse classification report
    report = parse_classification_report(text)
    if report:
        (out_dir / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"Saved classification_report.json -> {out_dir / 'classification_report.json'}")
    else:
        print("No classification report found in log.")

    # Parse and plot confusion matrix if present
    cm = parse_confusion_matrix_block(text)
    if cm is not None:
        class_names = args.class_names.split(',') if args.class_names else DEFAULT_CLASS_NAMES
        title = f"{args.model_name} - Confusion Matrix ({args.epoch_label})"
        plot_confusion_matrix(cm, class_names, title, out_dir / 'confusion_matrix.png')
        print(f"Saved confusion_matrix.png -> {out_dir / 'confusion_matrix.png'}")
    else:
        print("No numeric confusion matrix found in log.")


if __name__ == '__main__':
    main()
