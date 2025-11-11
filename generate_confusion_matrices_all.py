#!/usr/bin/env python3
"""
Generate Confusion Matrices for All Models (Epoch 10 and Epoch 20)
=================================================================
This script consolidates confusion matrix images for each model at epoch 10 and 20.
Priority order per model and epoch:
1) Parse the exact PNG path(s) from that model's training log (epochX/output trainig/*.txt) and copy them.
   - When copying, create an annotated version with a title banner so the image clearly shows the model and epoch.
2) If not found, look for model-specific files in epochX/saved_models_and_data and subfolders (and annotate).
3) If a numeric matrix is printed in the logs, reconstruct and save it with a proper title.
4) If JSON report(s) exist in parsed_outputs/epochX/** that contain 'confusion_matrix', render and save an image
   right next to each JSON (same folder as the report).

Output:
  ./confusion_matrices/
    ├── epoch10/
    └── epoch20/
Each containing {model}_confusion_matrix.png with a clear title for the model/epoch.
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# Root directory assumed to be this script's location
ROOT = Path(__file__).resolve().parent

# Epoch directories
EPOCH_DIRS = {
    "epoch10": ROOT / "epoch10",
    "epoch20": ROOT / "epoch20",
}

# Unified output directory
OUT_ROOT = ROOT / "confusion_matrices"

# Candidate filenames to look for per model (fallback only)
MODEL_CANDIDATES = {
    "convnext": [
        "convnext_confusion_matrix.png",
        "confusion_matrix.png",  # sometimes generic
    ],
    "sc_convnext": [
        "sc_convnext_confusion_matrix.png",
        "sc_convnext_confusion_matrix_high_res.png",
    ],
    "hybrid_cnn_vit": [
        "hybrid_cnn_vit_confusion_matrix.png",
        "hybrid_model_confusion_matrix.png",
    ],
    "hybrid_v2": [
        "hybrid_model_confusion_matrix.png",
    ],
    "yolo_efficientnet": [
        "hybrid_model_confusion_matrix.png",
    ],
    "protopnet": [
        "protopnet_confusion_matrix.png",
    ],
}

# Pretty names for titles
PRETTY_MODEL_NAME = {
    "convnext": "ConvNeXt",
    "sc_convnext": "SC-ConvNeXt",
    "hybrid_cnn_vit": "Hybrid CNN-ViT",
    "hybrid_v2": "Hybrid V2",
    "yolo_efficientnet": "YOLOv9 + EfficientNet-B3",
    "protopnet": "ProtoPNet",
}

# Class names (consistent across models per dataset logs)
CLASS_NAMES = [
    'aphid', 'army_worm', 'black_rust', 'brown_rust', 'common_rust',
    'fusarium_head_blight', 'healthy', 'leaf_blight', 'powdery_mildew_leaf',
    'spetoria', 'tan_spot', 'yellow_rust'
]

# Regex to capture saved confusion matrix paths in logs
SAVE_PATH_RE = re.compile(r"Confusion matrix saved to:?\s*(.+?\.png)")

# Map some log file name patterns to our canonical model keys
LOG_TO_MODEL_MAP = [
    (re.compile(r"convnext", re.I), "convnext"),
    (re.compile(r"sc\s*convnext|Train sc convnext", re.I), "sc_convnext"),
    (re.compile(r"hybrid\s*cnn\s*vit|hybrid cnn vit", re.I), "hybrid_cnn_vit"),
    (re.compile(r"hybrid\s*v2", re.I), "hybrid_v2"),
    (re.compile(r"yolo9|yolov9|efficient", re.I), "yolo_efficientnet"),
    (re.compile(r"protopnet", re.I), "protopnet"),
]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dest: Path) -> Optional[Path]:
    try:
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
            return dest
    except Exception as e:
        print(f"[ERROR] Copy failed {src} -> {dest}: {e}")
    return None


def annotate_image_with_title(img_path: Path, title_text: str) -> None:
    """Create an annotated version of img_path by adding a title banner at the top.
    The annotated image overwrites img_path.
    """
    try:
        img = Image.open(str(img_path)).convert("RGB")
        width, height = img.size
        banner_height = int(max(60, height * 0.08))
        banner = Image.new("RGB", (width, banner_height), color=(240, 240, 255))
        draw = ImageDraw.Draw(banner)
        # Try to load a default font; fallback to PIL default
        try:
            font = ImageFont.truetype("arial.ttf", size=int(banner_height * 0.45))
        except Exception:
            font = ImageFont.load_default()
        # Center text
        text_w, text_h = draw.textbbox((0, 0), title_text, font=font)[2:]
        draw.text(((width - text_w) // 2, (banner_height - text_h) // 2),
                  title_text, fill=(30, 30, 60), font=font)
        # Stack banner above original image
        out = Image.new("RGB", (width, banner_height + height), color=(255, 255, 255))
        out.paste(banner, (0, 0))
        out.paste(img, (0, banner_height))
        out.save(str(img_path), format="PNG")
    except Exception as e:
        print(f"[WARN] Could not annotate title for {img_path}: {e}")


def copy_if_exists(src_dir: Path, candidates: List[str], dest_path: Path) -> Optional[Path]:
    for name in candidates:
        cand = src_dir / name
        if cand.exists():
            return copy_file(cand, dest_path)
    return None


def parse_numeric_matrix_from_log(log_path: Path) -> Optional[np.ndarray]:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
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
        nums = [int(p) for p in parts if p]
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


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], model_key: str, epoch_name: str, save_path: Path) -> None:
    title = f"{PRETTY_MODEL_NAME.get(model_key, model_key)} - Confusion Matrix ({epoch_name})"
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


def resolve_class_names_from_json(data: dict) -> List[str]:
    # Prefer per_class keys order if provided; else fall back to default CLASS_NAMES
    per_class = data.get('per_class')
    if isinstance(per_class, dict) and per_class:
        return list(per_class.keys())
    return CLASS_NAMES


def pretty_title_for_json(data: dict, fallback_model_key: str, epoch_name: str) -> str:
    json_model = str(data.get('model', '') or '').strip()
    pretty = PRETTY_MODEL_NAME.get(json_model.lower(), None)
    if pretty:
        return f"{pretty} - Confusion Matrix ({epoch_name})"
    pretty2 = PRETTY_MODEL_NAME.get(fallback_model_key, None)
    if pretty2:
        return f"{pretty2} - Confusion Matrix ({epoch_name})"
    return f"{json_model or fallback_model_key} - Confusion Matrix ({epoch_name})"


def infer_model_key_from_log_name(name: str) -> Optional[str]:
    for pattern, key in LOG_TO_MODEL_MAP:
        if pattern.search(name):
            return key
    return None


def parse_saved_paths_from_logs(epoch_dir: Path) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    logs_dir = epoch_dir / "output trainig"
    if not logs_dir.exists():
        return result
    for txt in logs_dir.glob("*.txt"):
        model_key = infer_model_key_from_log_name(txt.name)
        if not model_key:
            continue
        text = txt.read_text(errors="ignore")
        paths = [Path(p.strip()) for p in SAVE_PATH_RE.findall(text)]
        if paths:
            result.setdefault(model_key, [])
            for p in paths:
                if not p.is_absolute():
                    p = (txt.parent / p).resolve()
                result[model_key].append(p)
    return result


def process_epoch(epoch_name: str) -> None:
    epoch_dir = EPOCH_DIRS[epoch_name]
    saved_dir = epoch_dir / "saved_models_and_data"
    out_dir = OUT_ROOT / epoch_name
    ensure_dir(out_dir)

    # 1) Try to use exact saved paths parsed from logs
    saved_by_logs = parse_saved_paths_from_logs(epoch_dir)

    for model_key in MODEL_CANDIDATES.keys():
        dest = out_dir / f"{model_key}_confusion_matrix.png"
        used = False
        if model_key in saved_by_logs:
            for src_path in saved_by_logs[model_key]:
                if src_path.exists():
                    copy_file(src_path, dest)
                    annotate_image_with_title(dest, f"{PRETTY_MODEL_NAME.get(model_key, model_key)} - Confusion Matrix ({epoch_name})")
                    print(f"[LOG] {model_key} -> {dest} (from {src_path})")
                    used = True
                    break
        if used:
            continue

        # 2) Fallback: look in saved_models_and_data and evaluation subfolders
        copied = copy_if_exists(saved_dir, MODEL_CANDIDATES[model_key], dest)
        if not copied:
            eval_dir = saved_dir / "evaluation_results"
            if eval_dir.exists():
                copied = copy_if_exists(eval_dir, MODEL_CANDIDATES[model_key], dest)
                if not copied:
                    copied = copy_if_exists(eval_dir / "heatmaps", MODEL_CANDIDATES[model_key], dest)
        if copied:
            annotate_image_with_title(dest, f"{PRETTY_MODEL_NAME.get(model_key, model_key)} - Confusion Matrix ({epoch_name})")
            print(f"[COPIED] {model_key} -> {copied}")
            continue

        # 3) Fallback: reconstruct from numeric matrix in logs
        logs_dir = epoch_dir / "output trainig"
        possible_logs = list(logs_dir.glob("*.txt")) if logs_dir.exists() else []
        chosen_log = None
        for txt in possible_logs:
            if infer_model_key_from_log_name(txt.name) == model_key:
                chosen_log = txt
                break
        if chosen_log and chosen_log.exists():
            cm = parse_numeric_matrix_from_log(chosen_log)
            if cm is not None:
                plot_confusion_matrix(cm, CLASS_NAMES, model_key, epoch_name, dest)
                print(f"[RECONSTRUCTED] {model_key} -> {dest}")
                continue

        # 4) Fallback: render from parsed JSON reports if they contain 'confusion_matrix'
        #    Save output in the SAME FOLDER as the JSON file (not consolidated folder)
        parsed_root = ROOT / "parsed_outputs" / epoch_name
        if parsed_root.exists():
            any_rendered = False
            for root_dir, _, files in os.walk(parsed_root):
                had_class_report = False
                rendered_in_dir = False
                for fn in files:
                    if not fn.lower().endswith('.json'):
                        continue
                    jf = Path(root_dir) / fn
                    if jf.name.lower() == 'classification_report.json':
                        had_class_report = True
                    try:
                        data = json.loads(jf.read_text(encoding='utf-8'))
                    except Exception:
                        continue
                    cm_list = data.get('confusion_matrix')
                    if not cm_list:
                        continue
                    try:
                        cm_np = np.array(cm_list, dtype=int)
                    except Exception:
                        continue
                    if cm_np.ndim != 2 or cm_np.shape[0] != cm_np.shape[1]:
                        continue
                    class_names = resolve_class_names_from_json(data)
                    # Choose model key based on folder name if possible, else fallback to provided model_key
                    folder_model = Path(root_dir).name.lower()
                    title = pretty_title_for_json(data, folder_model or model_key, epoch_name)
                    out_path = Path(root_dir) / 'confusion_matrix.png'
                    # Render image next to the JSON
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues',
                                xticklabels=class_names, yticklabels=class_names)
                    plt.title(title)
                    plt.xlabel('Predicted label')
                    plt.ylabel('True label')
                    plt.tight_layout()
                    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"[JSON->LOCAL] saved {out_path}")
                    rendered_in_dir = True
                    any_rendered = True
                if had_class_report and not rendered_in_dir:
                    print(f"[JSON->LOCAL] {root_dir} has classification_report.json but no confusion_matrix in any JSON; skipping.")
            if any_rendered:
                continue

        print(f"[MISSING] No confusion matrix found for {model_key} ({epoch_name})")


def main():
    ensure_dir(OUT_ROOT)
    for epoch_name in ("epoch10", "epoch20"):
        if not EPOCH_DIRS[epoch_name].exists():
            print(f"[WARN] {epoch_name} directory not found, skipping.")
            continue
        print(f"\n=== Processing {epoch_name} ===")
        process_epoch(epoch_name)
    print(f"\nDone. Consolidated confusion matrices in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
