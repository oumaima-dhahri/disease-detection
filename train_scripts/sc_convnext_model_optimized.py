import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import os

# -----------------------------
# Enhanced Channel Attention with SE-style
# -----------------------------
class EnhancedChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for efficiency
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# -----------------------------
# Enhanced Spatial Attention
# -----------------------------
class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.conv(x)

# -----------------------------
# Enhanced CBAM with Multiple Scales
# -----------------------------
class EnhancedCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(EnhancedCBAM, self).__init__()
        self.ca = EnhancedChannelAttention(in_planes, ratio)
        self.sa = EnhancedSpatialAttention(kernel_size)
        
    def forward(self, x):
        # Channel attention first
        x = x * self.ca(x)
        # Then spatial attention
        x = x * self.sa(x)
        return x

# -----------------------------
# Squeeze-and-Excitation Block
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# -----------------------------
# Multi-Scale Feature Fusion
# -----------------------------
class MultiScaleFusion(nn.Module):
    def __init__(self, channels):
        super(MultiScaleFusion, self).__init__()
        self.conv1x1 = nn.Conv2d(channels, channels, 1)
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(channels, channels, 5, padding=2)
        self.fusion = nn.Conv2d(channels * 3, channels, 1)
        
    def forward(self, x):
        # Multi-scale feature extraction
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        
        # Concatenate and fuse
        fused = torch.cat([f1, f3, f5], dim=1)
        return self.fusion(fused)

# -----------------------------
# OPTIMIZED SC-ConvNeXt with Multiple Attention Blocks
# -----------------------------
class OptimizedSCConvNeXt(nn.Module):
    def __init__(self, num_classes=12):
        super(OptimizedSCConvNeXt, self).__init__()
        
        # Load ConvNeXt with proper weights parameter
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # Multiple CBAM blocks at different stages for better feature refinement
        self.cbam1 = EnhancedCBAM(96)   # After stage 1 (96 channels)
        self.cbam2 = EnhancedCBAM(192)  # After stage 2 (192 channels)
        self.cbam3 = EnhancedCBAM(384)  # After stage 3 (384 channels)
        
        # SE blocks for additional attention
        self.se1 = SEBlock(96)
        self.se2 = SEBlock(192)
        self.se3 = SEBlock(384)
        
        # Multi-scale fusion blocks
        self.msf1 = MultiScaleFusion(96)
        self.msf2 = MultiScaleFusion(192)
        self.msf3 = MultiScaleFusion(384)
        
        # Feature pyramid network for better feature extraction
        self.fpn = nn.ModuleList([
            nn.Conv2d(96, 256, 1),   # Stage 1 -> 256
            nn.Conv2d(192, 256, 1),  # Stage 2 -> 256
            nn.Conv2d(384, 256, 1),  # Stage 3 -> 256
        ])
        
        # Global feature fusion
        self.global_fusion = nn.Sequential(
            nn.Conv2d(256 * 3, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features from each stage with attention
        # Stage 1
        x = self.backbone.features[0](x)  # Stem
        x = self.backbone.features[1](x)  # Stage 1
        x1 = self.cbam1(x)  # CBAM attention
        x1 = self.se1(x1)    # SE attention
        x1 = self.msf1(x1)   # Multi-scale fusion
        
        # Stage 2
        x = self.backbone.features[2](x)
        x2 = self.cbam2(x)  # CBAM attention
        x2 = self.se2(x2)    # SE attention
        x2 = self.msf2(x2)   # Multi-scale fusion
        
        # Stage 3
        x = self.backbone.features[3](x)
        x3 = self.cbam3(x)  # CBAM attention
        x3 = self.se3(x3)    # SE attention
        x3 = self.msf3(x3)   # Multi-scale fusion
        
        # Final stage
        x = self.backbone.features[4](x)
        
        # Feature Pyramid Network
        fpn1 = self.fpn[0](x1)  # 96 -> 256
        fpn2 = self.fpn[1](x2)  # 192 -> 256
        fpn3 = self.fpn[2](x3)  # 384 -> 256
        
        # Upsample to same size (use the largest feature map size)
        target_size = fpn3.shape[2:]
        fpn1 = F.interpolate(fpn1, size=target_size, mode='bilinear', align_corners=False)
        fpn2 = F.interpolate(fpn2, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        fused_features = torch.cat([fpn1, fpn2, fpn3], dim=1)  # 256*3 = 768 channels
        
        # Global fusion
        global_features = self.global_fusion(fused_features)  # 768 -> 512
        
        # Classification
        output = self.classifier(global_features)
        
        return output

# -----------------------------
# Advanced Focal Loss with Label Smoothing
# -----------------------------
class AdvancedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1, reduction='mean'):
        super(AdvancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1 - self.label_smoothing
            )
            smooth_targets += self.label_smoothing / num_classes
            targets = smooth_targets
        
        # Focal loss calculation
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------------
# Model Loading with SimCLR Support
# -----------------------------
def load_optimized_model(num_classes=12):
    """Load the optimized SC-ConvNeXt model"""
    model = OptimizedSCConvNeXt(num_classes=num_classes)
    
    # Try to load SimCLR weights if available
    simclr_weights = 'saved_models_and_data/simclr_convnext_tiny.pth'
    if os.path.exists(simclr_weights):
        print(f"Loading SimCLR pre-trained weights from {simclr_weights}")
        try:
            state_dict = torch.load(simclr_weights, map_location='cpu')
            # Remove classifier weights if present
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('classifier') and not k.startswith('fpn')}
            missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded SimCLR weights. Missing: {missing}, Unexpected: {unexpected}")
        except Exception as e:
            print(f"Error loading SimCLR weights: {e}")
            print("Continuing with ImageNet pre-trained weights")
    
    return model

# -----------------------------
# Model Summary Function
# -----------------------------
def print_model_summary(model):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture Summary:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Print attention blocks
    print(f"\nAttention Blocks:")
    print(f"- Enhanced CBAM blocks: 3")
    print(f"- SE blocks: 3")
    print(f"- Multi-scale fusion blocks: 3")
    print(f"- Feature Pyramid Network: 3 levels")

if __name__ == "__main__":
    # Test the model
    model = OptimizedSCConvNeXt(num_classes=12)
    print_model_summary(model)
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"\nTest forward pass:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model is working correctly! ✅")
