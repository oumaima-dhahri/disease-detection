"""
Inference + explanation script for ConvNeXt + Multi-Scale Fusion model.

Features:
  - Loads a trained checkpoint (.pth from convnext_msf_train_cam.py).
  - Runs evaluation on the test split.
  - Generates Grad-CAM overlays & gradient saliency maps for sample images.

Usage:
  python convnext_msf_inference_cam.py \
      --data-root dataset_split \
      --checkpoint outputs_msf_cam/convnext_msf.pth \
      --save-dir cam_results
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from convnext_msf_train_cam import (
    ConvNeXtMSF,
    build_transforms,
    compute_saliency,
    focal_loss,
    GradCAM,
    overlay_cam_on_image,
)


def build_test_loader(data_root: str, batch_size: int, num_workers: int = 4):
    _, eval_tf = build_transforms()
    test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=eval_tf)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, test_ds.classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = focal_loss(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


def save_saliency_plot(image_tensor: torch.Tensor, saliency: np.ndarray, save_path: Path):
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(saliency, cmap="inferno", alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ConvNeXt + MSF inference with Grad-CAM and saliency")
    parser.add_argument("--data-root", type=str, default="dataset_split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved .pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="cam_results")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of test samples for visualizations")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    loader, class_names = build_test_loader(args.data_root, args.batch_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    model = ConvNeXtMSF(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    test_loss, test_acc = evaluate(model, loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    with open(Path(args.save_dir) / "metrics.json", "w") as fp:
        json.dump({"test_loss": test_loss, "test_acc": test_acc}, fp, indent=2)

    model.eval()
    grad_cam = GradCAM(model, model.msf)

    sample_iter = iter(loader)
    images, labels = next(sample_iter)
    images, labels = images[: args.num_samples], labels[: args.num_samples]
    images = images.to(device)

    for idx, (img, label) in enumerate(zip(images, labels)):
        img_batch = img.unsqueeze(0)
        cam = grad_cam.generate(img_batch, target_class=label.item())
        overlay_path = Path(args.save_dir) / f"gradcam_overlay_{idx}.png"
        overlay_cam_on_image(img_batch, cam, overlay_path)
        saliency = compute_saliency(model, img_batch, target_class=label.item())
        save_saliency_plot(img_batch, saliency[0], Path(args.save_dir) / f"saliency_{idx}.png")

        torch.save(
            {
                "cam": cam,
                "label": class_names[label.item()],
            },
            Path(args.save_dir) / f"cam_{idx}.pt",
        )

    grad_cam.close()
    print(f"Saved Grad-CAM & saliency outputs to {args.save_dir}")


if __name__ == "__main__":
    main()

