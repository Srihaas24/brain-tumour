"""
Dataset and DataLoader utilities for BRISC Brain Tumor MRI Dataset.
Classes:
  - glioma
  - meningioma
  - no_tumor
  - pituitary
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMAGE_SIZE = (224, 224)

# Standard ImageNet normalization parameters
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_transforms():
    """Returns training and validation transforms."""
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

    return train_transform, val_transform

def get_dataloaders(data_dir="dataset/classification", batch_size=32, num_workers=0, subset_fraction=1.0):
    """
    Creates train and test DataLoaders from dataset/classification/train and dataset/classification/test.
    If subset_fraction < 1.0, randomly samples a subset for rapid experimentation.
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_tf, val_tf = get_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tf)

    if subset_fraction < 1.0:
        np.random.seed(42)
        train_indices = np.random.choice(
            len(train_dataset), 
            int(len(train_dataset) * subset_fraction), 
            replace=False
        )
        test_indices = np.random.choice(
            len(test_dataset), 
            int(len(test_dataset) * subset_fraction), 
            replace=False
        )
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, test_loader, CLASS_NAMES
