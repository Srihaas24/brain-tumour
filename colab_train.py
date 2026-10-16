"""
=============================================================================
Google Colab Standalone Brain Tumor Classifier Training Script
Dataset: BRISC 2025 (Automated download from Figshare - No Google Drive required)
Model: MobileNetV3 + Grad-CAM Explainability Hooks
=============================================================================
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models

# --- 1. CONFIGURATION ---
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMAGE_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# --- 2. DATA TRANSFORMS ---
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# --- 3. MODEL ARCHITECTURE ---
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4, backbone='mobilenet_v3'):
        super().__init__()
        self.num_classes = num_classes

        if backbone == 'mobilenet_v3':
            try:
                base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            except Exception:
                base = models.mobilenet_v3_small(weights=None)
            self.features = base.features
            in_features = base.classifier[0].in_features
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            self.target_layer = self.features[-1]
        elif backbone == 'resnet18':
            try:
                base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception:
                base = models.resnet18(weights=None)
            self.features = nn.Sequential(*list(base.children())[:-2])
            in_features = base.fc.in_features
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            self.target_layer = self.features[-1]

        # Grad-CAM storage
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)

# --- 4. TRAINING & EVALUATION FUNCTIONS ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# --- 5. MAIN EXECUTION ---
def main(epochs=10, batch_size=32, lr=0.0005, backbone='mobilenet_v3'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(f"🚀 Training Brain Tumor Classifier on {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backbone: {backbone} | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 65)

    # Locate dataset directory
    possible_paths = [
        "classification_task",
        "brisc2025/classification_task",
        "dataset/classification"
    ]
    data_dir = None
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "train")):
            data_dir = p
            break

    if data_dir is None:
        raise FileNotFoundError(
            "Could not find classification dataset folder. Did you unzip brisc2025.zip?"
        )

    print(f"Using dataset located at: {data_dir}")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    print(f"Classes: {train_ds.classes}")

    # Build model
    model = BrainTumorClassifier(num_classes=4, backbone=backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n--- Starting Training ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        ep_time = time.time() - ep_start
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'backbone': backbone,
                'class_names': CLASS_NAMES
            }, "best_model.pt")
            print(f"  ⭐ Best model updated (Val Acc: {val_acc*100:.2f}%) -> best_model.pt")

    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time/60:.2f} mins! Peak Test Accuracy: {best_val_acc*100:.2f}%")

    # Plot metrics
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()

if __name__ == '__main__':
    main(epochs=10, batch_size=32, lr=0.0005, backbone='mobilenet_v3')
