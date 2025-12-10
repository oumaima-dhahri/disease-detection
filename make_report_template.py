#!/usr/bin/env python3
"""
Make Report Template JSON
=========================
Creates a JSON template file you can fill in manually with per-class metrics
(precision, recall, f1_score, support), overall accuracy, and an optional
confusion matrix.

Usage:
  python make_report_template.py --out report.json \
    --model "ConvNeXt" --epoch epoch10 \
    --class-names "aphid,army_worm,black_rust,brown_rust,common_rust,fusarium_head_blight,healthy,leaf_blight,powdery_mildew_leaf,spetoria,tan_spot,yellow_rust"

The template includes placeholders (null/0) for all values. Fill them in manually.
"""

import argparse
import json
from pathlib import Path

DEFAULT_CLASS_NAMES = [
    'aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf',
    'spetoria', 'tan_spot', 'yellow_rust'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='Path to output JSON file')
    parser.add_argument('--model', default='Model', help='Model name to include in JSON')
    parser.add_argument('--epoch', default='epoch', help='Epoch label (e.g., epoch10)')
    parser.add_argument('--class-names', default=None, help='Comma-separated class names; defaults to 12 wheat disease classes')
    args = parser.parse_args()

    class_names = [c.strip() for c in (args.class_names.split(',') if args.class_names else DEFAULT_CLASS_NAMES)]

    data = {
        'model': args.model,
        'epoch': args.epoch,
        'overall': {
            'accuracy': None,    # fill manually (e.g., 0.9147)
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
        # Optional: 2D list of ints (12x12) after you paste numbers
        'confusion_matrix': None
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"Template written to: {out_path}")

if __name__ == '__main__':
    main()
















