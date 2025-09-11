import os
import subprocess
import sys
from pathlib import Path

def run_fixed_script():
    """Run the fixed script"""
    
    script_file = "train_model_yolov9_efficient_net_fixed.py"
    
    # Check if script exists
    if not Path(script_file).is_file():
        print(f"❌ Script '{script_file}' not found!")
        return 1
    
    print(f"🚀 Running {script_file}...")
    print("This script has all the problematic code fixed!")
    print("=" * 60)
    
    try:
        # Run the script with output
        process = subprocess.Popen(
            [sys.executable, script_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
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
        
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return 1

def run_fixed_silent():
    """Run the fixed script silently"""
    
    script_file = "train_model_yolov9_efficient_net_fixed.py"
    
    # Check if script exists
    if not Path(script_file).is_file():
        print(f"❌ Script '{script_file}' not found!")
        return 1
    
    print(f"🚀 Running {script_file} silently...")
    print("All outputs will be saved to files. No logs will be shown.")
    
    try:
        # Run the script silently
        process = subprocess.Popen(
            [sys.executable, script_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
        
    except Exception as e:
        print(f"❌ Error running script: {e}")
        return 1

if __name__ == "__main__":
    print("Choose execution mode:")
    print("1. Run with output (recommended for first run)")
    print("2. Run silently (no output)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        exit_code = run_fixed_silent()
    else:
        exit_code = run_fixed_script()
    
    sys.exit(exit_code)
