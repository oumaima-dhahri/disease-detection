#!/usr/bin/env python3
"""
Script to suppress PyTorch tensor normalization warnings.
Run this before executing your training scripts.
"""

import warnings
import os

def suppress_pytorch_warnings():
    """Suppress common PyTorch warnings"""
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Suppress specific PyTorch warnings
    warnings.filterwarnings('ignore', message='.*torch.Tensor inputs should be normalized.*')
    warnings.filterwarnings('ignore', message='.*Dividing input by 255.*')
    warnings.filterwarnings('ignore', message='.*User provided device.*')
    warnings.filterwarnings('ignore', message='.*User provided device type.*')
    
    # Suppress matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # Suppress seaborn warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
    
    # Set environment variables to suppress some warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    print("✅ PyTorch warnings suppressed successfully!")
    print("You can now run your training scripts without warning messages.")

if __name__ == "__main__":
    suppress_pytorch_warnings()
    
    # Example usage
    print("\nExample usage:")
    print("1. Run this script first:")
    print("   python suppress_warnings.py")
    print("\n2. Then run your training script:")
    print("   python train_yolo_v9.py")
    print("\n3. Or import this in your script:")
    print("   from suppress_warnings import suppress_pytorch_warnings")
    print("   suppress_pytorch_warnings()")
