"""
Training Script for Brain Tumor MRI Classification.
Trains a Convolutional Neural Network on the BRISC 2025 dataset.
"""

import os
import sys
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

from dataset import get_dataloaders, CLASS_NAMES
from model import BrainTumorClassifier, TUMOR_INFO




def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_metrics(history, output_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#2563eb', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='#dc2626', linewidth=2, linestyle='--')
    plt.title('Cross-Entropy Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='#16a34a', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='#9333ea', linewidth=2, linestyle='--')
    plt.title('Classification Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_training(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print("=" * 60)
    print("🧠 BRISC Brain Tumor Classifier Training Pipeline")
    print(f"Device: {device} | Backbone: {args.backbone} | Epochs: {args.epochs}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 60)

    os.makedirs(args.model_dir, exist_ok=True)

    # 1. Load Data
    print("\n[1/4] Preparing Dataset & DataLoaders...")
    train_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        subset_fraction=args.subset
    )
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # 2. Build Model
    print(f"\n[2/4] Initializing {args.backbone} model (pretrained={args.pretrained})...")
    model = BrainTumorClassifier(num_classes=len(class_names), pretrained=args.pretrained, backbone=args.backbone)
    model = model.to(device)


    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training Loop
    print("\n[3/4] Starting Training Loop...")
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        ep_duration = time.time() - ep_start
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.model_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'backbone': args.backbone,
                'class_names': class_names
            }, best_model_path)
            print(f" ⭐ New best model saved! (Val Acc: {val_acc*100:.2f}%) -> {best_model_path}")

    total_time = time.time() - start_time
    print(f"\n[4/4] Training Complete in {total_time/60:.2f} minutes! Best Val Acc: {best_val_acc*100:.2f}%")

    # Plot metrics
    metrics_plot_path = os.path.join(args.model_dir, "training_curves.png")
    plot_metrics(history, metrics_plot_path)
    print(f"📊 Training curves saved to {metrics_plot_path}")

    # Save metadata JSON
    metadata = {
        'backbone': args.backbone,
        'best_val_acc': float(best_val_acc),
        'epochs': args.epochs,
        'classes': class_names,
        'image_size': [224, 224],
        'total_training_time_sec': round(total_time, 2),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    metadata_path = os.path.join(args.model_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Metadata saved to {metadata_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Brain Tumor Classifier")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--backbone', type=str, default='mobilenet_v3', choices=['mobilenet_v3', 'resnet18', 'custom'], help="Backbone architecture")
    parser.add_argument('--pretrained', action='store_true', default=False, help="Attempt to download online ImageNet weights")
    parser.add_argument('--data-dir', type=str, default='dataset/classification', help="Path to classification directory")

    parser.add_argument('--model-dir', type=str, default='models', help="Directory to save model weights")
    parser.add_argument('--subset', type=float, default=1.0, help="Fraction of dataset to use (1.0 = 100%)")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to use")
    
    args = parser.parse_args()
    run_training(args)
