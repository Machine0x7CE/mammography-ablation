"""
Dataset and data loading utilities for mammography classification.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from .config import TrainingConfig


class MammographyDataset(Dataset):
    """Custom dataset for mammography images."""
    
    def __init__(self, root_dir: str, transform=None, split: str = 'train'):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing train/test folders
            transform: Image transformations to apply
            split: 'train' or 'test'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.split = split
        
        self.image_folder = ImageFolder(self.root_dir, transform=None)
        self.classes = self.image_folder.classes
        self.class_to_idx = self.image_folder.class_to_idx
        self.samples = self.image_folder.samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load as grayscale and convert to RGB for pretrained models
        image = Image.open(img_path).convert('L')
        image = Image.merge('RGB', (image, image, image))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate balanced class weights for imbalanced data.
        
        Formula: Weight(Class) = Total_Samples / (Num_Classes * Count(Class))
        
        This is the "balanced" weighting strategy used by sklearn.
        
        Why this works:
        - Weights are inversely proportional to class frequency
        - For balanced data (50/50), both weights = 1.0
        - For imbalanced data, minority class gets weight > 1
        - This makes the loss function penalize errors on minority class more
        
        Example with our mammography data:
        - Total = 2810, Benign = 1641, Malignant = 1169
        - Benign weight = 2810 / (2 * 1641) = 0.857 (majority, lower weight)
        - Malignant weight = 2810 / (2 * 1169) = 1.202 (minority, higher weight)
        
        This is CRITICAL for cancer detection:
        - We want to avoid missing malignant cases (false negatives)
        - Higher weight on malignant means model is penalized more for missing cancer
        """
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        num_classes = len(class_counts)
        
        # Balanced class weights: total / (num_classes * count_per_class)
        weights = total_samples / (num_classes * class_counts)
        
        return torch.FloatTensor(weights)


def get_train_transforms():
    """Get training data transformations with augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_test_transforms():
    """Get test data transformations (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_data_loaders(config: TrainingConfig):
    """
    Create training and test data loaders.
    
    Args:
        config: Training configuration
    
    Returns:
        train_loader, test_loader, class_weights
    """
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    train_dataset = MammographyDataset(
        config.data_root, 
        transform=train_transform, 
        split='train'
    )
    test_dataset = MammographyDataset(
        config.data_root, 
        transform=test_transform, 
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    class_weights = train_dataset.get_class_weights()
    
    return train_loader, test_loader, class_weights


def print_data_info(train_loader, test_loader, class_weights):
    """Print dataset information including class distribution and weights."""
    train_dataset = train_loader.dataset
    
    # Count samples per class
    labels = [label for _, label in train_dataset.samples]
    benign_count = labels.count(0)
    malignant_count = labels.count(1)
    total = len(labels)
    
    print(f"\nDataset Summary:")
    print(f"  Training samples: {total}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    print(f"\nClass Distribution (Training Set):")
    print(f"  Benign (0):    {benign_count:>5} samples ({100*benign_count/total:.1f}%)")
    print(f"  Malignant (1): {malignant_count:>5} samples ({100*malignant_count/total:.1f}%)")
    
    print(f"\nBalanced Class Weights:")
    print(f"  Benign weight:    {class_weights[0]:.4f} (majority class, lower weight)")
    print(f"  Malignant weight: {class_weights[1]:.4f} (minority class, higher weight)")
    print(f"\n  -> Errors on malignant samples penalized {class_weights[1]/class_weights[0]:.2f}x more than benign")

