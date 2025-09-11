import os
import json
import subprocess
import sys
import re
from pathlib import Path

def diagnose_notebook_issues(notebook_path):
    """Diagnose issues in the notebook"""
    
    print(f"🔍 Diagnosing {notebook_path}...")
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Check for problematic patterns
    issues_found = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                cell_code = ''.join(source)
            else:
                cell_code = source
            
            # Check for problematic patterns
            problematic_patterns = [
                ('get_ipython()', 'get_ipython()'),
                ('%matplotlib', '%matplotlib'),
                ('plt.show()', 'plt.show()'),
                ('IPython.display', 'IPython.display'),
                ('sys.exit', 'sys.exit'),
                ('SystemExit', 'SystemExit'),
                ('%tb', '%tb'),
            ]
            
            for pattern, description in problematic_patterns:
                if pattern in cell_code:
                    issues_found.append(f"Cell {i}: Found {description}")
    
    if issues_found:
        print("❌ Issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("✅ No obvious issues found in notebook")
    
    return issues_found

def create_minimal_test_script(notebook_path):
    """Create a minimal test script to isolate the issue"""
    
    print("🧪 Creating minimal test script...")
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Extract just the imports and basic setup
    minimal_code = """
import os
import sys
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

print("✅ Basic imports successful")

# Test basic functionality
try:
    print("Testing PyTorch...")
    x = torch.randn(2, 3)
    print(f"PyTorch tensor created: {x.shape}")
    
    print("Testing matplotlib...")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    plt.close()
    print("✅ Matplotlib test successful")
    
    print("Testing ultralytics...")
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics import successful")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
    
    print("✅ All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
"""
    
    # Write minimal test file
    test_file = "minimal_test.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(minimal_code)
    
    print(f"✅ Minimal test script created: {test_file}")
    return test_file

def run_minimal_test(test_file):
    """Run the minimal test to identify the issue"""
    
    print(f"🚀 Running {test_file}...")
    print("=" * 50)
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print("=" * 50)
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Minimal test passed!")
            return True
        else:
            print("❌ Minimal test failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def check_system_requirements():
    """Check system requirements"""
    
    print("🔍 Checking system requirements...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Available memory: {memory.available / (1024**3):.1f} GB")
        print(f"Total memory: {memory.total / (1024**3):.1f} GB")
    except ImportError:
        print("psutil not available for memory check")
    
    # Check CUDA availability
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
    except ImportError:
        print("PyTorch not available")

if __name__ == "__main__":
    notebook_file = "train_model_yolov9_efficient_net.ipynb"
    
    # Check if notebook exists
    if not Path(notebook_file).is_file():
        print(f"❌ Notebook '{notebook_file}' not found!")
        sys.exit(1)
    
    print("🔍 DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Check system requirements
    check_system_requirements()
    print()
    
    # Diagnose notebook issues
    issues = diagnose_notebook_issues(notebook_file)
    print()
    
    # Create and run minimal test
    test_file = create_minimal_test_script(notebook_file)
    print()
    
    success = run_minimal_test(test_file)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ System appears to be working correctly")
        print("The issue might be in the notebook conversion or specific code")
    else:
        print("❌ System has issues that need to be resolved")
        print("Check the error messages above for details")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"Cleaned up {test_file}")
