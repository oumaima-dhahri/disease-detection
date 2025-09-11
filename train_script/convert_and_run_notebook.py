import os
import subprocess
import json
from pathlib import Path

# === CONFIG ===
NOTEBOOK_FILE = "train_model_yolov9_efficient_net.ipynb"  # change to any notebook
OUTPUT_PY_FILE = NOTEBOOK_FILE.replace(".ipynb", ".py")
# =================

print(f"📁 Current directory: {os.getcwd()}")
print("🔍 Contents:")
for f in sorted(os.listdir(".")):
    print(f"  → {f}")

# Check notebook exists
if not Path(NOTEBOOK_FILE).is_file():
    raise FileNotFoundError(f"Notebook '{NOTEBOOK_FILE}' not found!")

# Step 1: Convert notebook to Python script (manual conversion since nbconvert may not be available)
print(f"\n📝 Converting '{NOTEBOOK_FILE}' → '{OUTPUT_PY_FILE}' ...")

try:
    # Try using jupyter nbconvert first
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "script", NOTEBOOK_FILE],
        check=True,
        capture_output=True
    )
    print("✅ Conversion done using jupyter nbconvert!")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("jupyter nbconvert not available, using manual conversion...")
    
    # Manual conversion
    with open(NOTEBOOK_FILE, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Extract code from all cells
    code_lines = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                code_lines.extend(source)
            else:
                code_lines.append(source)

    # Join all code lines
    full_code = ''.join(code_lines)

    # Write to Python file
    with open(OUTPUT_PY_FILE, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    print("✅ Manual conversion completed!")

# Step 2: Check the converted script for any remaining issues
print(f"\n🔍 Checking converted script...")
with open(OUTPUT_PY_FILE, 'r', encoding='utf-8') as f:
    content = f.read()
    
# Check for problematic patterns
issues = []
if '%matplotlib' in content:
    issues.append('Found %matplotlib magic command')
if 'get_ipython()' in content:
    issues.append('Found get_ipython() call')
if 'plt.show()' in content:
    issues.append('Found plt.show() calls')
if 'IPython.display' in content:
    issues.append('Found IPython.display import')

if issues:
    print('❌ Issues found in converted script:')
    for issue in issues:
        print(f'  - {issue}')
    print("\nPlease fix these issues before running the script.")
else:
    print('✅ No issues found in converted script!')
    
    # Step 3: Run the script and stream output live
    print(f"\n🚀 Running '{OUTPUT_PY_FILE}' ...\n" + "━" * 60)
    process = subprocess.Popen(
        ["python", OUTPUT_PY_FILE],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream logs line by line
    for line in process.stdout:
        print(line, end="")

    process.wait()
    print("\n🎉 Script finished! All training logs should be shown above.")
