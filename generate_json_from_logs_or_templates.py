#!/usr/bin/env python3
"""
Generate JSONs From Logs Or Templates
=====================================
Scans epoch10/epoch20 output logs. For each model log:
- If a classification report is found, parse it to JSON.
- If a numeric confusion matrix block is present, include it in the JSON.
- If no report is found, create a manual template JSON to fill.

Outputs mirror: parsed_outputs/<epoch>/<model_key>/
  - classification_report.json (parsed) OR report_template.json (manual)
  - confusion_matrix.png can be rendered later with render_confusion_from_json.py
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
EPOCH_DIRS = [ROOT / 'epoch10', ROOT / 'epoch20']
OUT_ROOT = ROOT / 'parsed_outputs'

DEFAULT_CLASS_NAMES = [
    'aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf',
    'spetoria', 'tan_spot', 'yellow_rust'
]

LOG_TO_MODEL_MAP = [
    (re.compile(r"convnext", re.I), "convnext", "ConvNeXt"),
    (re.compile(r"sc\s*convnext|Train sc convnext", re.I), "sc_convnext", "SC-ConvNeXt"),
    (re.compile(r"hybrid\s*cnn\s*vit|hybrid cnn vit", re.I), "hybrid_cnn_vit", "Hybrid CNN-ViT"),
    (re.compile(r"hybrid\s*v2", re.I), "hybrid_v2", "Hybrid V2"),
    (re.compile(r"yolo9|yolov9|efficient", re.I), "yolo_efficientnet", "YOLOv9 + EfficientNet-B3"),
    (re.compile(r"protopnet", re.I), "protopnet", "ProtoPNet"),
]


def infer_model_key_and_name(filename: str):
    for patt, key, pretty in LOG_TO_MODEL_MAP:
        if patt.search(filename):
            return key, pretty
    return None, None


def parse_classification_report(text: str) -> Optional[Dict]:
    lines = text.splitlines()
    parsed: Dict[str, Dict[str, float]] = {}
    started = False
    any_row = False
    for line in lines:
        raw = line.strip()
        if not raw:
            if started and any_row:
                break
            else:
                continue
        low = raw.lower()
        if not started and ('precision' in low and 'recall' in low and 'f1' in low and 'support' in low):
            started = True
            continue
        if not started:
            continue
        parts = re.split(r"\s+", raw)
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


def parse_confusion_matrix(text: str) -> Optional[List[List[int]]]:
    m = re.search(r"Confusion matrix values:\s*\n(\[.+?\])", text, flags=re.S)
    block = m.group(1) if m else None
    if not block:
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
    return mat.tolist()


def write_template(out_path: Path, model_name: str, epoch_label: str, class_names: List[str]) -> None:
    data = {
        'model': model_name,
        'epoch': epoch_label,
        'overall': {
            'accuracy': None,
            'macro_avg_f1': None,
            'weighted_avg_f1': None,
            'support': None
        },
        'per_class': {
            name: {
                'precision': None,
                'recall': None,
                'f1_score': None,
                'support': None
            } for name in class_names
        },
        'confusion_matrix': None
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    for epoch_dir in EPOCH_DIRS:
        if not epoch_dir.exists():
            continue
        logs_dir = epoch_dir / 'output trainig'
        if not logs_dir.exists():
            continue
        for log_file in logs_dir.glob('*.txt'):
            model_key, pretty_name = infer_model_key_and_name(log_file.name)
            if not model_key:
                continue
            epoch_label = epoch_dir.name  # 'epoch10' or 'epoch20'
            out_dir = OUT_ROOT / epoch_label / model_key
            out_dir.mkdir(parents=True, exist_ok=True)

            text = log_file.read_text(errors='ignore')
            report = parse_classification_report(text)
            cm = parse_confusion_matrix(text)

            if report:
                # Save parsed report
                (out_dir / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                # If confusion matrix found, embed alongside in combined JSON too
                combined = {
                    'model': pretty_name,
                    'epoch': epoch_label,
                    'per_class': {k: v for k, v in report.items() if k not in ('accuracy',)},
                    'overall': report.get('accuracy', {}),
                    'confusion_matrix': cm
                }
                (out_dir / 'report_with_optional_cm.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
                print(f"[PARSED] {epoch_label}/{model_key} -> classification_report.json (+ optional CM)")
            else:
                # Create manual template if report not present
                write_template(out_dir / 'report_template.json', pretty_name, epoch_label, DEFAULT_CLASS_NAMES)
                print(f"[TEMPLATE] {epoch_label}/{model_key} -> report_template.json (fill manually)")

if __name__ == '__main__':
    main()













