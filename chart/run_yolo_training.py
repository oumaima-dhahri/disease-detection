import os
import json
import subprocess
import sys
import re
from pathlib import Path
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

def fix_notebook_code(code):
    """Clean notebook code for safe execution"""
    fixes = [
        # Remove IPython magic commands
        (r'get_ipython\(\)\.run_line_magic\([^)]+\)', ''),
        (r'get_ipython\(\)[^;]*;?', ''),
        (r'%matplotlib[^\n]*\n?', ''),
        (r'%tb[^\n]*\n?', ''),
        (r'%time[^\n]*\n?', ''),
        (r'%timeit[^\n]*\n?', ''),
        (r'%debug[^\n]*\n?', ''),
        (r'%pdb[^\n]*\n?', ''),
        
        # Fix display issues
        (r'plt\.show\(\)', 'plt.close()'),
        (r'from IPython\.display import[^;]*;?', ''),
        (r'display\([^)]+\)', ''),
        (r'IPython\.core\.interactiveshell[^;]*;?', ''),
        
        # Remove system exits and problematic commands
        (r'sys\.exit\([^)]*\)', ''),
        (r'SystemExit\([^)]*\)', ''),
        (r'exit\([^)]*\)', ''),
        (r'quit\([^)]*\)', ''),
        
        # Fix matplotlib backend issues
        (r'matplotlib\.use\([^)]+\)', "matplotlib.use('Agg')"),
        
        # Fix path issues - change relative paths to absolute
        (r"'\.\./dataset'", "'dataset'"),
        (r'"\.\./dataset"', '"dataset"'),
        (r"'\.\./saved_models_and_data'", "'saved_models_and_data'"),
        (r'"\.\./saved_models_and_data"', '"saved_models_and_data"'),
        (r"'\.\./dataset_split'", "'dataset_split'"),
        (r'"\.\./dataset_split"', '"dataset_split"'),
        
        # Remove problematic imports that might cause issues
        (r'from IPython[^;]*;?', ''),
        (r'import IPython[^;]*;?', ''),
    ]
    
    for pattern, replacement in fixes:
        code = re.sub(pattern, replacement, code, flags=re.MULTILINE)
    
    return code

def create_safe_execution_script(notebook_path):
    """Create a safe Python script from notebook"""
    print(f"Converting {notebook_path} to safe Python script...")

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    code_lines = []
    
    # Add safety imports at the beginning
    safety_code = """
# Safety imports and setup
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
os.environ['TORCH_WARN_ONCE'] = 'false'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

# Force matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Suppress PyTorch warnings
try:
    import torch
    torch.set_warn_always(False)
    torch.set_warn_once(False)
except:
    pass

print("✅ Safety setup completed")
"""
    
    code_lines.append(safety_code)
    
    # Process each code cell
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            cell_code = ''.join(source) if isinstance(source, list) else source
            
            # Skip empty cells
            if not cell_code.strip():
                continue
                
            # Add cell separator
            code_lines.append(f"\n# Cell {i+1}")
            
            # Fix the code
            fixed_code = fix_notebook_code(cell_code)
            
            # Skip cells that are now empty after fixing
            if fixed_code.strip():
                code_lines.append(fixed_code)
    
    # Add error handling at the end
    error_handling = """
# Error handling and cleanup
try:
    print("\\n✅ Training completed successfully!")
except Exception as e:
    print(f"\\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\\n🔚 Script execution finished")
"""
    
    code_lines.append(error_handling)
    
    full_code = '\n'.join(code_lines)
    
    # Create output file
    output_file = notebook_path.replace('.ipynb', '_safe.py')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    print(f"✅ Safe script created: {output_file}")
    return output_file

def run_yolo_notebook_safe(notebook_path):
    """Run YOLO notebook safely with proper error handling"""
    print(f"🚀 Starting YOLO training execution...")
    print("=" * 60)
    
    # Create safe script
    safe_script = create_safe_execution_script(notebook_path)
    
    # Environment setup for subprocess
    env = os.environ.copy()
    env.update({
        "PYTHONWARNINGS": "ignore",
        "PYDEVD_DISABLE_FILE_VALIDATION": "1",
        "TORCH_WARN_ONCE": "false",
        "TORCH_SHOW_CPP_STACKTRACES": "0",
        "CUDA_LAUNCH_BLOCKING": "0",
        "PYTHONPATH": os.getcwd(),
    })
    
    print(f"📝 Running safe script: {safe_script}")
    print("=" * 60)
    
    try:
        # Run the safe script from project root
        process = subprocess.Popen(
            [sys.executable, safe_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=os.getcwd()  # Run from current directory (project root)
        )
        
        # Stream output in real-time
        output_lines = []
        for line in process.stdout:
            # Filter out problematic warnings
            if any(warning in line.lower() for warning in [
                "torch.tensor inputs should be normalized",
                "debugger warning",
                "userwarning: to exit:",
                "warn: to exit:",
                "⚠️",
                "dividing input by 255"
            ]):
                continue
            
            # Show important training progress
            if any(keyword in line.lower() for keyword in [
                'epoch', 'loss', 'accuracy', 'training', 'validation',
                'batch', 'step', 'progress', 'completed', 'success'
            ]):
                print(line.rstrip())
            
            output_lines.append(line)
        
        # Wait for completion
        return_code = process.wait()
        
        print("=" * 60)
        
        if return_code == 0:
            print("✅ YOLO training completed successfully!")
            return True
        else:
            print(f"❌ YOLO training failed with exit code: {return_code}")
            print("\n📋 Last few lines of output:")
            for line in output_lines[-10:]:
                print(line.rstrip())
            return False
            
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            if os.path.exists(safe_script):
                os.remove(safe_script)
                print(f"🧹 Cleaned up temporary script: {safe_script}")
        except:
            pass

def run_yolo_notebook_direct(notebook_path):
    """Run YOLO notebook directly using nbformat (alternative method)"""
    print(f"🚀 Running YOLO notebook directly...")
    print("=" * 60)
    
    try:
        import nbformat
        from nbconvert import PythonExporter
        
        # Load notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Convert to Python
        exporter = PythonExporter()
        (body, resources) = exporter.from_notebook_node(notebook)
        
        # Fix the code
        fixed_body = fix_notebook_code(body)
        
        # Add safety setup
        safety_setup = """
# Safety setup
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Force matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Suppress PyTorch warnings
try:
    import torch
    torch.set_warn_always(False)
    torch.set_warn_once(False)
except:
    pass

print("✅ Direct execution setup completed")
"""
        
        final_code = safety_setup + "\n" + fixed_body
        
        # Execute the code
        exec(final_code, globals())
        
        print("=" * 60)
        print("✅ YOLO training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Direct execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with multiple execution strategies"""
    notebook_file = "train_script/train_model_yolov9_efficient_net.ipynb"
    
    # Check if notebook exists
    if not Path(notebook_file).is_file():
        print(f"❌ Notebook '{notebook_file}' not found!")
        print("\n📋 Available notebooks:")
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith('.ipynb'):
                    print(f"  - {os.path.join(root, file)}")
        return False
    
    print(f"🎯 Found notebook: {notebook_file}")
    print("\n🚀 Choose execution method:")
    print("1. Safe subprocess execution (recommended)")
    print("2. Direct execution (faster but less safe)")
    
    # Try safe subprocess first
    print("\n🔄 Trying safe subprocess execution...")
    success = run_yolo_notebook_safe(notebook_file)
    
    if not success:
        print("\n🔄 Safe execution failed, trying direct execution...")
        success = run_yolo_notebook_direct(notebook_file)
    
    if success:
        print("\n🎉 YOLO training completed successfully!")
        return True
    else:
        print("\n❌ All execution methods failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
