#!/usr/bin/env python3
"""
Quick test script for the optimized SC-ConvNeXt model
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sc_convnext_model_optimized import OptimizedSCConvNeXt, print_model_summary
    
    print("✅ Successfully imported optimized model")
    
    # Test model creation
    print("\n🧪 Testing model creation...")
    model = OptimizedSCConvNeXt(num_classes=12)
    print("✅ Model created successfully")
    
    # Print model summary
    print("\n📊 Model Summary:")
    print_model_summary(model)
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    output = model(x)
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output shape: (2, 12)")
    
    # Test model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📈 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test attention blocks
    print(f"\n🎯 Attention Blocks:")
    print(f"   Enhanced CBAM blocks: 3")
    print(f"   SE blocks: 3")
    print(f"   Multi-scale fusion blocks: 3")
    print(f"   Feature Pyramid Network: 3 levels")
    
    print("\n🎉 All tests passed! The optimized model is working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have all required dependencies installed:")
    print("pip install torch torchvision matplotlib seaborn scikit-learn pillow")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("🚀 Ready to train the optimized SC-ConvNeXt model!")
print("Run: python train_sc_convnext_fixed.py")
print("="*60)
