#!/usr/bin/env python3
"""
🚀 KAGGLE NOTEBOOK EXECUTION SCRIPT
This script will execute the fixed notebook and actually run the training
"""

import os
import json
import builtins

# === ✅ Set notebook to inspect ===
NOTEBOOK_FILE = "train_sc_convnext.ipynb"
# =================================

# Safely override print for real-time output
if 'original_print' not in globals():
    original_print = builtins.print

def flushed_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)

builtins.print = flushed_print

print(f"📁 Current directory: {os.getcwd()}")
print("🔍 Contents:")
for item in sorted(os.listdir(".")):
    print(f"  → {item}")

# Check if notebook exists
print(f"\n🔍 Loading: {NOTEBOOK_FILE}")
if not os.path.exists(NOTEBOOK_FILE):
    print(f"❌ Not found! Did you upload it?")
    exit(1)

# Load notebook
try:
    with open(NOTEBOOK_FILE, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"✅ Loaded '{NOTEBOOK_FILE}' with {len(nb['cells'])} cells.\n")
except Exception as e:
    print(f"❌ Failed to load: {type(e).__name__}: {e}")
    exit(1)

# Now execute ALL cells including the main() call
print("🚀 Executing cells...\n" + "━" * 60)
global_ns = {}

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source']).strip()
    if not source:
        continue

    print(f"⚡ [Cell {idx + 1}]")
    print(f"{'─' * 40}")

    try:
        exec(source, global_ns)
        print("✅ Executed.")
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    print(f"{'━' * 60}")

print(f"\n🎉 Execution finished. Check if training started above.")
