"""
Evaluation script for trained Brain Tumor Classifier.
Computes Confusion Matrix, Precision, Recall, F1-Score, and overall Test Accuracy.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dataset import get_dataloaders, CLASS_NAMES
from model import BrainTumorClassifier, CLASS_DISPLAY



def evaluate_model(model_path="models/best_model.pt", data_dir="dataset/classification", output_dir="models"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("🔬 Brain Tumor Model Evaluation & Diagnostic Report")
    print(f"Loading checkpoint: {model_path} on {device}")
    print("=" * 60)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}. Train the model first!")

    checkpoint = torch.load(model_path, map_location=device)
    backbone = checkpoint.get('backbone', 'mobilenet_v3')
    class_names = checkpoint.get('class_names', CLASS_NAMES)

    model = BrainTumorClassifier(num_classes=len(class_names), pretrained=False, backbone=backbone)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    _, test_loader, _ = get_dataloaders(data_dir=data_dir, batch_size=32)

    all_preds = []
    all_targets = []

    print("\nRunning inference across the test split...")
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Classification Report
    display_names = [CLASS_DISPLAY.get(c, c) for c in class_names]
    report = classification_report(all_targets, all_preds, target_names=display_names, digits=4)
    print("\n📋 Detailed Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=display_names, yticklabels=display_names, cbar=False)
    plt.title(f'Confusion Matrix — Test Set (Overall Acc: {np.mean(all_preds == all_targets)*100:.2f}%)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('Ground Truth Label', fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"📊 Confusion matrix visualization saved to {cm_path}")

    # Save metrics JSON
    metrics_summary = {
        'overall_accuracy': float(np.mean(all_preds == all_targets)),
        'total_test_samples': int(len(all_targets)),
        'confusion_matrix': cm.tolist(),
        'classes': class_names
    }
    with open(os.path.join(output_dir, "test_evaluation.json"), 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n✅ Evaluation successfully completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Classifier")
    parser.add_argument('--model-path', type=str, default='models/best_model.pt', help="Path to checkpoint")
    parser.add_argument('--data-dir', type=str, default='dataset/classification', help="Path to classification directory")
    parser.add_argument('--output-dir', type=str, default='models', help="Directory to save evaluation artifacts")

    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.output_dir)
