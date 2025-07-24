import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image
import random
import numpy as np

# -----------------------------
# SimCLR Dataset (ignores labels)
# -----------------------------
class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    self.samples.append(img_path)
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            xi = self.transform(image)
            xj = self.transform(image)
            return xi, xj
        return image, image

# -----------------------------
# SimCLR Augmentations
# -----------------------------
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# SimCLR Model
# -----------------------------
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, projection_dim)
        )
    def forward(self, x):
        features = self.encoder(x)
        if isinstance(features, (list, tuple)):
            features = features[0]
        features = torch.flatten(features, 1)
        z = self.projection_head(features)
        return z

# -----------------------------
# NT-Xent Loss (Contrastive Loss)
# -----------------------------
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = nn.functional.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)
    labels = torch.arange(batch_size).to(z.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)
    positives = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=-1)
    loss = -torch.log(positives / denominator)
    return loss.mean()

# -----------------------------
# Main SimCLR Pre-training
# -----------------------------
def main():
    DATASET_DIR = 'dataset'
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_PATH = 'saved_models_and_data/simclr_convnext_tiny.pth'

    dataset = SimCLRDataset(DATASET_DIR, transform=simclr_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # Encoder: ConvNeXt-T without classifier
    base_encoder = convnext_tiny(pretrained=True)
    base_encoder.classifier = nn.Identity()
    base_encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model = SimCLR(base_encoder, projection_dim=128).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('Starting SimCLR pre-training...')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xi, xj in loader:
            xi, xj = xi.to(DEVICE), xj.to(DEVICE)
            zi = model(xi)
            zj = model(xj)
            loss = nt_xent_loss(zi, zj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.encoder.state_dict(), SAVE_PATH)
            print(f"Saved encoder weights at epoch {epoch+1}")
    torch.save(model.encoder.state_dict(), SAVE_PATH)
    print(f"SimCLR pre-training complete. Encoder weights saved to {SAVE_PATH}")

if __name__ == "__main__":
    main() 