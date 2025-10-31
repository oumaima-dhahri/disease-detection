#!/usr/bin/env python3
"""
Grad-CAM Visualization for Top 3 Models
Models: ConvNeXt, SC-ConvNeXt, Hybrid CNN-ViT
Epochs: 10 and 20
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timm

# Configuration
IMAGE_SIZE = (224, 224)
ALPHA = 0.5  # Heatmap overlay transparency

# Paths
EPOCH10_MODELS_DIR = "epoch10/saved_models_and_data"
EPOCH20_MODELS_DIR = "epoch20/saved_models_and_data"
TEST_IMAGES_DIR = "epoch20/test_images"
DATASET_DIR = "dataset"

# Get class labels
def get_class_labels(dataset_dir):
    return sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

class_labels = get_class_labels(DATASET_DIR)
print(f"Classes: {class_labels}")

# Image preprocessing
test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

de_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


class ConvNeXtModel(nn.Module):
    """ConvNeXt Model Architecture"""
    def __init__(self, num_classes):
        super().__init__()
        self.convnext = models.convnext_tiny(pretrained=False)
        convnext_out = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(convnext_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.convnext.features(x)
        features = features.mean(dim=[2, 3])
        out = self.classifier(features)
        return out


# CBAM classes for SC-ConvNeXt
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class SCConvNeXtModel(nn.Module):
    """SC-ConvNeXt Model Architecture"""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.convnext_tiny(pretrained=False)
        self.cbam = CBAM(384)
        
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(384, eps=1e-6),
            nn.Linear(384, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone.features[0](x)
        x = self.backbone.features[1](x)
        x = self.backbone.features[2](x)
        x = self.backbone.features[3](x)
        x = self.backbone.features[4](x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier(x)
        return x


class HybridCNNViT(nn.Module):
    """Hybrid CNN-ViT Model Architecture"""
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.convnext_base(pretrained=False)
        cnn_out = self.cnn.classifier[2].in_features
        self.cnn.classifier = nn.Identity()
        
        self.vit = create_model('vit_base_patch16_224', pretrained=False)
        vit_out = self.vit.head.in_features
        self.vit.head = nn.Identity()
        
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + vit_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        cnn_feat = self.cnn.features(x)
        cnn_feat = cnn_feat.mean(dim=[2, 3])
        vit_feat = self.vit(x)
        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        out = self.fusion(fused)
        return out


class OptimizedGradCAM:
    """Optimized Grad-CAM implementation with proper gradient flow"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))
    
    def __call__(self, input_tensor, class_idx=None):
        # Enable gradients on input
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # Shape: [C, H, W]
        activations = self.activations[0]  # Shape: [C, H, W]
        
        # Compute importance weights using global average pooling
        weights = gradients.mean(dim=(1, 2))  # Shape: [C]
        
        # Create CAM by weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU to get only positive contributions
        cam = torch.relu(cam)
        
        # Convert to numpy and process
        cam = cam.detach().cpu().numpy()
        
        # Apply Gaussian smoothing
        cam = cv2.GaussianBlur(cam, (15, 15), 0)
        
        # Normalization
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        
        # Apply threshold to focus on important regions
        threshold = 0.15
        cam[cam < threshold] = 0
        
        # Resize to original image size
        cam = cv2.resize(cam, IMAGE_SIZE)
        
        return cam
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


def get_target_layer(model, model_type):
    """Get the target layer for Grad-CAM based on model type"""
    if model_type == "convnext":
        return model.convnext.features[-2]
    elif model_type == "sc_convnext":
        return model.backbone.features[-2]
    elif model_type == "hybrid":
        return model.cnn.features[-2]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def overlay_heatmap(img, heatmap):
    """Overlay heatmap on image"""
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    overlayed = ALPHA * img + (1 - ALPHA) * heatmap
    overlayed = np.clip(overlayed, 0, 1)
    return overlayed


def visualize_gradcam_for_model(model, model_name, epoch, test_images, device):
    """Visualize Grad-CAM for a specific model"""
    print(f"\n{'='*80}")
    print(f"Visualizing Grad-CAM for: {model_name} (Epoch {epoch})")
    print(f"{'='*80}")
    
    # Get model type
    if "ConvNeXt" in model_name and "SC" not in model_name:
        model_type = "convnext"
    elif "SC-ConvNeXt" in model_name or "SCConvNeXt" in model_name:
        model_type = "sc_convnext"
    elif "Hybrid" in model_name:
        model_type = "hybrid"
    else:
        print(f"Unknown model type: {model_name}")
        return
    
    # Get target layer
    target_layer = get_target_layer(model, model_type)
    gradcam = OptimizedGradCAM(model, target_layer)
    
    # Create output directory
    output_dir = f"gradcam_results/{model_name}_epoch{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, img_name in enumerate(test_images):
        print(f"\nProcessing image {i+1}/{len(test_images)}: {img_name}")
        
        try:
            # Load and preprocess image
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)
            img_pil = Image.open(img_path).convert('RGB')
            img_np = np.array(img_pil.resize(IMAGE_SIZE))
            img_tensor = test_transform(img_pil).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)[0]
                pred_idx = prob.argmax().item()
                pred_label = class_labels[pred_idx]
                pred_prob = prob[pred_idx].item()
            
            print(f"Prediction: {pred_label} (Confidence: {pred_prob:.3f})")
            
            # Generate Grad-CAM
            cam = gradcam(img_tensor, class_idx=pred_idx)
            
            # Normalize image for overlay
            img_norm = img_np.astype(np.float32) / 255.0
            
            # Create overlay
            cam_overlay = overlay_heatmap(img_norm, cam)
            
            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(img_pil)
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(cam_overlay)
            axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            axes[3].axis('off')
            axes[3].text(0.5, 0.7, f'Model: {model_name}\nEpoch: {epoch}\n',
                        ha='center', va='top', fontsize=12, fontweight='bold')
            axes[3].text(0.5, 0.5, f'Prediction: {pred_label}\nConfidence: {pred_prob:.3f}',
                        ha='center', va='center', fontsize=12)
            axes[3].text(0.5, 0.3, f'Image: {img_name}',
                        ha='center', va='bottom', fontsize=10)
            
            plt.suptitle(f'{model_name} (Epoch {epoch}) - {img_name}', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(output_dir, f"{img_name.replace('.', '_')}_gradcam.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            results.append({
                'image': img_name,
                'prediction': pred_label,
                'confidence': pred_prob
            })
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Clean up
    gradcam.remove_hooks()
    
    print(f"\n✅ Saved {len(results)} visualizations to {output_dir}")
    return results


def load_model(model_name, epoch, device):
    """Load a specific model"""
    if epoch == 10:
        models_dir = EPOCH10_MODELS_DIR
    else:
        models_dir = EPOCH20_MODELS_DIR
    
    num_classes = len(class_labels)
    
    if "ConvNeXt" in model_name and "SC" not in model_name:
        # Standard ConvNeXt
        model = ConvNeXtModel(num_classes)
        model_path = os.path.join(models_dir, "wheat_disease_convnext_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, "best_convnext_model.pth")
    elif "SC-ConvNeXt" in model_name or "SCConvNeXt" in model_name:
        # SC-ConvNeXt
        model = SCConvNeXtModel(num_classes)
        model_path = os.path.join(models_dir, "wheat_disease_sc_convnext_model.pth")
    elif "Hybrid" in model_name:
        # Hybrid CNN-ViT
        model = HybridCNNViT(num_classes)
        model_path = os.path.join(models_dir, "final_hybrid_model.pth")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        return None
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded {model_name} from {model_path}")
    return model


def main():
    """Main execution function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get test images
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.jfif'))]
    
    print(f"Found {len(test_images)} test images")
    
    # Models to visualize
    models_to_visualize = [
        ("ConvNeXt", 10),
        ("ConvNeXt", 20),
        ("SC-ConvNeXt", 10),
        ("SC-ConvNeXt", 20),
        ("Hybrid CNN-ViT", 10),
        ("Hybrid CNN-ViT", 20),
    ]
    
    all_results = {}
    
    for model_name, epoch in models_to_visualize:
        print(f"\n{'#'*80}")
        print(f"PROCESSING: {model_name} (Epoch {epoch})")
        print(f"{'#'*80}")
        
        # Load model
        model = load_model(model_name, epoch, device)
        if model is None:
            continue
        
        # Generate visualizations
        results = visualize_gradcam_for_model(model, model_name, epoch, test_images, device)
        all_results[f"{model_name}_epoch{epoch}"] = results
    
    # Summary
    print(f"\n{'='*80}")
    print("GRAD-CAM VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total models processed: {len(all_results)}")
    print(f"Total images processed: {len(test_images)}")
    print(f"\nResults saved in: gradcam_results/")
    
    # Count total visualizations
    total_viz = sum(len(r) for r in all_results.values())
    print(f"Total visualizations generated: {total_viz}")


if __name__ == "__main__":
    main()





