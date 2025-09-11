import os
import json
import subprocess
import sys
import re
from pathlib import Path

def fix_notebook_code(code):
    """Fix problematic code patterns in notebook"""
    
    # Remove problematic patterns
    fixes = [
        # Remove get_ipython() calls and magic commands
        (r'get_ipython\(\)\.run_line_magic\([^)]+\)', ''),
        (r'get_ipython\(\)[^;]*;?', ''),
        (r'%matplotlib[^\n]*\n?', ''),
        (r'%tb[^\n]*\n?', ''),
        
        # Replace plt.show() with plt.close()
        (r'plt\.show\(\)', 'plt.close()'),
        
        # Remove IPython.display imports and calls
        (r'from IPython\.display import[^;]*;?', ''),
        (r'display\([^)]+\)', ''),
        
        # Fix any remaining IPython references
        (r'IPython\.core\.interactiveshell[^;]*;?', ''),
        
        # Remove any SystemExit calls
        (r'sys\.exit\([^)]*\)', ''),
        (r'SystemExit\([^)]*\)', ''),
    ]
    
    for pattern, replacement in fixes:
        code = re.sub(pattern, replacement, code)
    
    return code

def run_notebook_debug(notebook_path):
    """Run notebook with debug output to see what's happening"""
    
    print(f"Converting {notebook_path} to Python script...")
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Extract and fix code from all cells
    code_lines = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                cell_code = ''.join(source)
            else:
                cell_code = source
            
            # Fix problematic code
            fixed_code = fix_notebook_code(cell_code)
            code_lines.append(fixed_code)

    # Join all code lines
    full_code = '\n'.join(code_lines)
    
    # Additional safety fixes
    print("Applying safety fixes...")
    
    # Ensure matplotlib backend is set correctly
    if 'matplotlib.use(' not in full_code:
        full_code = "import matplotlib\nmatplotlib.use('Agg')\n" + full_code
    
    # Remove any remaining problematic imports and calls
    full_code = re.sub(r'from IPython[^\n]*\n', '', full_code)
    full_code = re.sub(r'import IPython[^\n]*\n', '', full_code)
    full_code = re.sub(r'sys\.exit\([^)]*\)', '', full_code)
    
    # Write fixed Python file
    output_file = notebook_path.replace('.ipynb', '_debug.py')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    print("✅ Notebook converted and fixed")
    print(f"Running {output_file} with debug output...")
    print("=" * 60)
    
    # Run the fixed script with debug output
    process = subprocess.Popen(
        [sys.executable, output_file],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout
        text=True,
        bufsize=1
    )
    
    # Show output line by line
    for line in process.stdout:
        print(line, end="")
    
    # Wait for completion
    return_code = process.wait()
    
    print("=" * 60)
    if return_code == 0:
        print("✅ Training completed successfully!")
        print("Check the 'saved_models_and_data' directory for outputs.")
    else:
        print(f"❌ Training failed with exit code: {return_code}")
    
    return return_code

if __name__ == "__main__":
    # Default notebook
    notebook_file = "train_model_yolov9_efficient_net.ipynb"
    
    # Check if notebook exists
    if not Path(notebook_file).is_file():
        print(f"❌ Notebook '{notebook_file}' not found!")
        print("Available notebooks:")
        for f in sorted(os.listdir(".")):
            if f.endswith('.ipynb'):
                print(f"  - {f}")
        sys.exit(1)
    
    # Run the notebook with debug
    exit_code = run_notebook_debug(notebook_file)
    sys.exit(exit_code)
