import os
import subprocess
import sys
from pathlib import Path

def run_yolo_script():
    """Run the existing yolo.py script silently"""
    
    script_file = "yolo.py"
    
    # Check if script exists
    if not Path(script_file).is_file():
        print(f"❌ Script '{script_file}' not found!")
        print("Available Python files:")
        for f in sorted(os.listdir(".")):
            if f.endswith('.py'):
                print(f"  - {f}")
        return 1
    
    print(f"Running {script_file} silently...")
    print("All outputs will be saved to files. No logs will be shown.")
    
    # Run the script silently
    process = subprocess.Popen(
        [sys.executable, script_file],
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
    
    return return_code

def run_yolo_debug():
    """Run the existing yolo.py script with debug output"""
    
    script_file = "yolo.py"
    
    # Check if script exists
    if not Path(script_file).is_file():
        print(f"❌ Script '{script_file}' not found!")
        print("Available Python files:")
        for f in sorted(os.listdir(".")):
            if f.endswith('.py'):
                print(f"  - {f}")
        return 1
    
    print(f"Running {script_file} with debug output...")
    print("=" * 60)
    
    # Run the script with debug output
    process = subprocess.Popen(
        [sys.executable, script_file],
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
    print("Choose execution mode:")
    print("1. Silent execution (no output)")
    print("2. Debug execution (show output)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        exit_code = run_yolo_debug()
    else:
        exit_code = run_yolo_script()
    
    sys.exit(exit_code)
