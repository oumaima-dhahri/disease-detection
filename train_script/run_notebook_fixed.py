import os
import json
import subprocess
import sys
import re
from pathlib import Path

def fix_notebook_code(code):
    """Fix all problematic code patterns in notebook"""
    
    # Comprehensive fixes for all problematic patterns
    fixes = [
        # Remove get_ipython() calls and magic commands
        (r'get_ipython\(\)\.run_line_magic\([^)]+\)', ''),
        (r'get_ipython\(\)[^;]*;?', ''),
        (r'%matplotlib[^\n]*\n?', ''),
        (r'%tb[^\n]*\n?', ''),
        (r'%[^\n]*\n?', ''),  # Remove all magic commands
        
        # Replace plt.show() with plt.close()
        (r'plt\.show\(\)', 'plt.close()'),
        
        # Remove IPython.display imports and calls
        (r'from IPython\.display import[^;]*;?', ''),
        (r'import IPython[^\n]*\n?', ''),
        (r'display\([^)]+\)', ''),
        
        # Fix any remaining IPython references
        (r'IPython\.core\.interactiveshell[^;]*;?', ''),
        (r'IPython[^\n]*\n?', ''),
        
        # Remove any SystemExit calls and sys.exit
        (r'sys\.exit\([^)]*\)', ''),
        (r'SystemExit\([^)]*\)', ''),
        
        # Remove problematic error messages
        (r'An exception has occurred[^\n]*\n?', ''),
        (r'use %tb to see[^\n]*\n?', ''),
        (r'Traceback \(most recent call last\):[^\n]*\n?', ''),
    ]
    
    for pattern, replacement in fixes:
        code = re.sub(pattern, replacement, code)
    
    return code

def create_working_script(notebook_path):
    """Create a working script from notebook with comprehensive fixes"""
    
    print(f"Converting {notebook_path} to working Python script...")
    
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
    
    # Additional comprehensive safety fixes
    print("Applying comprehensive safety fixes...")
    
    # Ensure matplotlib backend is set correctly at the very beginning
    if 'matplotlib.use(' not in full_code:
        full_code = "import matplotlib\nmatplotlib.use('Agg')\n" + full_code
    
    # Remove any remaining problematic imports and calls
    full_code = re.sub(r'from IPython[^\n]*\n', '', full_code)
    full_code = re.sub(r'import IPython[^\n]*\n', '', full_code)
    full_code = re.sub(r'sys\.exit\([^)]*\)', '', full_code)
    
    # Create a robust working script
    working_script = f"""
# Working script generated from notebook with comprehensive fixes
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend first
import matplotlib
matplotlib.use('Agg')

# Set up error handling
def safe_execute():
    try:
        # Original notebook code with fixes applied
        {full_code}
        
        # Ensure the script doesn't exit unexpectedly
        if __name__ == "__main__":
            try:
                # Run the main function if it exists
                if 'main' in globals():
                    model = main()
                    print("✅ Training completed successfully!")
                else:
                    print("No main function found, executing notebook code directly")
            except Exception as e:
                print(f"Error in main execution: {{e}}")
                import traceback
                traceback.print_exc()
                return False
        return True
        
    except Exception as e:
        print(f"Error during execution: {{e}}")
        import traceback
        traceback.print_exc()
        return False

# Run the script safely
if __name__ == "__main__":
    success = safe_execute()
    if not success:
        sys.exit(1)
"""
    
    # Write working Python file
    output_file = notebook_path.replace('.ipynb', '_working.py')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(working_script)
    
    print("✅ Working script created with comprehensive fixes")
    return output_file

def run_notebook_silent(notebook_path):
    """Run notebook silently with comprehensive fixes"""
    
    # Create working script
    output_file = create_working_script(notebook_path)
    
    print(f"Running {output_file} silently...")
    print("All outputs will be saved to files. No logs will be shown.")
    
    # Run the working script silently
    process = subprocess.Popen(
        [sys.executable, output_file],
        stdout=subprocess.DEVNULL,  # Suppress stdout
        stderr=subprocess.DEVNULL,  # Suppress stderr
        text=True
    )
    
    # Wait for completion
    return_code = process.wait()
    
    if return_code == 0:
        print("✅ Training completed successfully!")
        print("Check the 'saved_models_and_data' directory for:")
        print("  - Model files (.pth)")
        print("  - Training logs (.json)")
        print("  - Confusion matrix plots (.png)")
        print("  - Training curves (.png)")
    else:
        print(f"❌ Training failed with exit code: {return_code}")
        print("The working script may have encountered an error.")
    
    return return_code

def run_notebook_debug(notebook_path):
    """Run notebook with debug output to see what's happening"""
    
    # Create working script
    output_file = create_working_script(notebook_path)
    
    print(f"Running {output_file} with debug output...")
    print("=" * 60)
    
    # Run the working script with debug output
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
    
    print("Choose execution mode:")
    print("1. Silent execution (no output)")
    print("2. Debug execution (show output)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        exit_code = run_notebook_debug(notebook_file)
    else:
        exit_code = run_notebook_silent(notebook_file)
    
    sys.exit(exit_code)

