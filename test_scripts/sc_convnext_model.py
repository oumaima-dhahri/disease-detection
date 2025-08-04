import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny

# -----------------------------
# Improved CBAM with LeakyReLU
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.leaky_relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.leaky_relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.leaky_relu(self.conv1(x))
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# -----------------------------
# ConvNeXt-T + CBAM
# -----------------------------
class SCConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super(SCConvNeXt, self).__init__()
        self.backbone = convnext_tiny(pretrained=True)
        # Only one CBAM after the final stage
        self.cbam = CBAM(384)
        # Replace classifier to match 384 channels
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),  # (batch, 384, 1, 1) -> (batch, 384)
            nn.LayerNorm(384, eps=1e-6),
            nn.Linear(384, num_classes)
        )
    def forward(self, x):
        # Stem and stages
        x = self.backbone.features[0](x)
        x = self.backbone.features[1](x)
        x = self.backbone.features[2](x)
        x = self.backbone.features[3](x)
        x = self.backbone.features[4](x)
        x = self.cbam(x)  # Only after the last stage
        x = self.backbone.avgpool(x)
        x = self.backbone.classifier(x)
        return x
    def load_simclr_weights(self, simclr_weights_path):
        """
        Load SimCLR pre-trained weights into the ConvNeXt-T backbone.
        Usage:
            model = SCConvNeXt(num_classes)
            model.load_simclr_weights('saved_models_and_data/simclr_convnext_tiny.pth')
        """
        state_dict = torch.load(simclr_weights_path, map_location='cpu')
        # Remove classifier weights if present
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded SimCLR weights. Missing: {missing}, Unexpected: {unexpected}")

# -----------------------------
# Focal Loss
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
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
# SimCLR Pre-training Placeholder
# -----------------------------
# To use SimCLR pre-training, you can implement or import a SimCLR module here.
# For now, this is a placeholder. You can pre-train the backbone using SimCLR and load the weights before supervised training.
# Example:
# from simclr_module import SimCLRPretrainer
# simclr = SimCLRPretrainer(self.backbone)
# simclr.pretrain(...) 