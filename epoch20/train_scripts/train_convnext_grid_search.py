import os
import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image


# -----------------------------
# Default configuration
# -----------------------------

DATASET_DIR = "../dataset"
SAVE_DIR = "../saved_models_and_data"
SPLIT_OUTPUT_DIR = "../dataset_split"
IMAGE_SIZE = (224, 224)
TEST_SIZE = 0.15
VAL_SIZE = 0.15
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, test_transform


# -----------------------------
# Dataset (same logic as notebook)
# -----------------------------


class WheatDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for img_file in os.listdir(class_dir):
                path = os.path.join(class_dir, img_file)
                self.samples.append((path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, target
        except Exception as e:
            print(f"Erreur lors du chargement de {path}: {e}")
            return self.__getitem__((idx + 1) % len(self))


def get_data_loaders(batch_size: int):
    train_transform, test_transform = build_transforms()

    split_dirs = [
        os.path.join(SPLIT_OUTPUT_DIR, split) for split in ["train", "val", "test"]
    ]
    split_exists = all(os.path.isdir(d) and len(os.listdir(d)) > 0 for d in split_dirs)

    if split_exists:
        print("Found existing split dataset. Loading splits...")
        train_dataset = WheatDiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, "train"), transform=train_transform
        )
        val_dataset = WheatDiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, "val"), transform=test_transform
        )
        test_dataset = WheatDiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, "test"), transform=test_transform
        )
    else:
        print("No split dataset found. Splitting and saving images...")
        full_dataset = WheatDiseaseDataset(DATASET_DIR, transform=train_transform)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(full_dataset), generator=generator).tolist()
        train_size = int((1 - TEST_SIZE - VAL_SIZE) * len(full_dataset))
        val_size = int(VAL_SIZE * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]
        train_data = Subset(full_dataset, train_indices)
        val_data = Subset(full_dataset, val_indices)
        test_data = Subset(full_dataset, test_indices)

        def save_split_images(dataset, indices, split_name):
            print(f"Saving images for split: {split_name}")
            for idx in indices:
                path, label_idx = dataset.dataset.samples[idx]  # dataset is a Subset
                class_name = dataset.dataset.classes[label_idx]
                filename = os.path.basename(path)
                dest_dir = os.path.join(SPLIT_OUTPUT_DIR, split_name, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
                if not os.path.exists(dest_path):
                    try:
                        import shutil

                        shutil.copyfile(path, dest_path)
                    except Exception as e:
                        print(f"Failed to copy {path} to {dest_path}: {e}")

        save_split_images(train_data, train_indices, "train")
        save_split_images(val_data, val_indices, "val")
        save_split_images(test_data, test_indices, "test")
        print("Image splits saved.")
        train_dataset = WheatDiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, "train"), transform=train_transform
        )
        val_dataset = WheatDiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, "val"), transform=test_transform
        )
        test_dataset = WheatDiseaseDataset(
            os.path.join(SPLIT_OUTPUT_DIR, "test"), transform=test_transform
        )

    print("Calculating class weights for balanced sampling...")
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    print("Data loaders are ready.")
    return train_loader, val_loader, test_loader, train_dataset.classes


# -----------------------------
# Model & training (as in notebook, but parametric)
# -----------------------------


def load_model(num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = models.convnext_base(pretrained=True)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    return model, device


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    trial_name: str,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)
    best_acc = 0.0
    no_improvement_epochs = 0
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)
    train_log: List[Dict] = []

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{trial_name}] Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {time.time()-start_time:.1f}s"
        )

        train_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc.item(),
                "val_loss": val_loss,
                "val_acc": val_acc.item(),
                "lr": current_lr,
            }
        )

        # Early Stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(
                    SAVE_DIR, f"best_convnext_model_{trial_name.replace(' ', '_')}.pth"
                ),
            )
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"[{trial_name}] Early stopping triggered.")
            break

    return model, train_log, best_acc.item()


def evaluate_on_test(model, device, test_loader, class_labels):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_labels, digits=4, output_dict=True
    )
    return conf_matrix, report


# -----------------------------
# Simple grid search
# -----------------------------


def run_grid_search(
    learning_rates: List[float],
    batch_sizes: List[int],
    num_epochs: int = EPOCHS,
):
    results: List[Dict] = []
    trial_id = 0

    for lr, batch_size in product(learning_rates, batch_sizes):
        trial_id += 1
        trial_name = f"trial{trial_id}_lr{lr}_bs{batch_size}"
        print("\n" + "=" * 80)
        print(f"Starting {trial_name}")
        print("=" * 80)

        train_loader, val_loader, test_loader, class_labels = get_data_loaders(
            batch_size=batch_size
        )
        model, device = load_model(len(class_labels))

        model, train_log, best_val_acc = train_model(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=lr,
            trial_name=trial_name,
        )

        # Load best weights for this trial
        best_model_path = os.path.join(
            SAVE_DIR, f"best_convnext_model_{trial_name.replace(' ', '_')}.pth"
        )
        if os.path.exists(best_model_path):
            model.load_state_dict(
                torch.load(best_model_path, map_location=device)
            )

        conf_matrix, report = evaluate_on_test(
            model, device, test_loader, class_labels
        )

        # Compute overall test accuracy from report
        test_acc = report["accuracy"]

        trial_result = {
            "trial_name": trial_name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
        }
        results.append(trial_result)

        # Save per-trial logs
        trial_dir = Path(SAVE_DIR) / "grid_search_convnext" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        with open(trial_dir / "train_log.json", "w") as f:
            json.dump(train_log, f, indent=2)
        with open(trial_dir / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        np.save(trial_dir / "confusion_matrix.npy", conf_matrix)

    # Sort results by validation accuracy
    results_sorted = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
    leaderboard_path = Path(SAVE_DIR) / "grid_search_convnext" / "leaderboard.json"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(leaderboard_path, "w") as f:
        json.dump(results_sorted, f, indent=2)

    print("\nGrid search completed. Top configurations:")
    for r in results_sorted[:5]:
        print(
            f"{r['trial_name']} | lr={r['learning_rate']} | "
            f"bs={r['batch_size']} | best_val_acc={r['best_val_acc']:.4f} | "
            f"test_acc={r['test_acc']:.4f}"
        )


if __name__ == "__main__":
    # You can edit these lists to try different values
    lr_list = [1e-4, 5e-5]
    batch_size_list = [16, 32]
    run_grid_search(learning_rates=lr_list, batch_sizes=batch_size_list, num_epochs=EPOCHS)

