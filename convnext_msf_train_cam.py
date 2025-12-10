"""
ConvNeXt + Multi-Scale Fusion training script with Grad-CAM and saliency maps.

Usage:
    python convnext_msf_train_cam.py --data-root dataset_split --epochs 25

This script:
    - Builds a ConvNeXt-Base backbone with the existing 3-branch MSF head.
    - Trains using focal loss + MixUp-ready pipeline (MixUp optional).
    - Saves Grad-CAM heatmaps, raw saliency maps, and Grad-CAM overlays for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -------------------------------------------------------------
# Model definition (ConvNeXt backbone + Multi-Scale Fusion head)
# -------------------------------------------------------------


class MultiScaleFusion(nn.Module):
    """Three-branch grouped convolution fusion block."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        branch = lambda k: nn.Sequential(
            nn.Conv2d(channels, channels, k, padding=k // 2, groups=max(1, channels // groups)),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch3 = branch(3)
        self.branch5 = branch(5)
        self.branch7 = branch(7)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        fused = self.fusion(torch.cat([b3, b5, b7], dim=1))
        return fused + x


class ConvNeXtMSF(nn.Module):
    """ConvNeXt backbone with multi-scale fusion and classification head."""

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        self.stem = backbone.features
        self.msf = MultiScaleFusion(1024)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.stem(x)
        return self.msf(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.classifier(feats)


# ---------------
# Training utils
# ---------------


def build_transforms(image_size: int = 320):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def get_dataloaders(data_root: str, batch_size: int, num_workers: int = 4):
    train_tf, eval_tf = build_transforms()
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds.classes


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = focal_loss(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = focal_loss(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


# -----------------------------------
# Grad-CAM and saliency visualisation
# -----------------------------------


class GradCAM:
    """Simple Grad-CAM implementation hooking a target layer."""

    def __init__(self, model: ConvNeXtMSF, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        loss = logits[:, target_class].mean()
        loss.backward()

        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        activations = self.activations
        cam = F.relu((gradients * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze().cpu().numpy()

    def close(self):
        for handle in self.hook_handles:
            handle.remove()


@torch.no_grad()
def compute_saliency(model: ConvNeXtMSF, image: torch.Tensor, target_class: Optional[int] = None):
    """Simple gradient-based saliency map."""
    image = image.clone().detach().requires_grad_(True)
    logits = model(image)
    if target_class is None:
        target_class = logits.argmax(dim=1)
    score = logits[0, target_class]
    score.backward()
    saliency = image.grad.abs().max(dim=1)[0]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.cpu().numpy()


def overlay_cam_on_image(image_tensor: torch.Tensor, cam: np.ndarray, output_path: Path):
    """Creates Grad-CAM overlay and saves to disk."""
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# -------
# Driver
# -------


def main():
    parser = argparse.ArgumentParser(description="ConvNeXt + MSF training with Grad-CAM")
    parser.add_argument("--data-root", type=str, default="dataset_split", help="Root folder with train/val/test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="outputs_msf_cam")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--generate-cam", action="store_true", help="Generate Grad-CAM and saliency samples after training")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(args.data_root, args.batch_size)

    model = ConvNeXtMSF(num_classes=len(class_names)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Train {train_loss:.3f}/{train_acc:.3f} | Val {val_loss:.3f}/{val_acc:.3f}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": class_names,
            "history": history,
        },
        os.path.join(args.output_dir, "convnext_msf.pth"),
    )

    with open(os.path.join(args.output_dir, "history.json"), "w") as fp:
        json.dump(history, fp, indent=2)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test loss {test_loss:.3f} | Test acc {test_acc:.3f}")

    if args.generate_cam:
        model.eval()
        grad_cam = GradCAM(model, model.msf)
        sample_images, sample_labels = next(iter(test_loader))
        sample_images = sample_images[:4].to(device)
        sample_labels = sample_labels[:4]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, (image, label) in enumerate(zip(sample_images, sample_labels)):
            image = image.unsqueeze(0)
            cam = grad_cam.generate(image, target_class=label.item())
            saliency = compute_saliency(model, image)
            overlay_path = Path(args.output_dir) / f"grad_cam_overlay_{timestamp}_{idx}.png"
            overlay_cam_on_image(image, cam, overlay_path)
            np.save(Path(args.output_dir) / f"grad_cam_raw_{timestamp}_{idx}.npy", cam)
            np.save(Path(args.output_dir) / f"saliency_{timestamp}_{idx}.npy", saliency)
        grad_cam.close()


if __name__ == "__main__":
    main()

