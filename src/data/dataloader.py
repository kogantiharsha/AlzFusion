"""
Data loading utilities for multi-modal Alzheimer's dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import io


class MultiModalDataset(Dataset):
    """
    Dataset class for multi-modal Alzheimer's data.
    Handles both genetic variant and MRI image data.
    """
    
    def __init__(self, genetic_data, mri_data, labels=None, transform=None):
        """
        Args:
            genetic_data: NumPy array of genetic features (N, 130)
            mri_data: DataFrame with 'image' column containing image bytes
            labels: NumPy array of labels (N,)
            transform: Optional transform to apply to images
        """
        self.genetic_data = genetic_data
        self.mri_data = mri_data
        self.labels = labels
        self.transform = transform
        
        # Ensure data alignment
        assert len(genetic_data) == len(mri_data), \
            f"Genetic data ({len(genetic_data)}) and MRI data ({len(mri_data)}) must have same length"
        
        if labels is not None:
            assert len(genetic_data) == len(labels), \
                f"Data ({len(genetic_data)}) and labels ({len(labels)}) must have same length"
    
    def __len__(self):
        return len(self.genetic_data)
    
    def __getitem__(self, idx):
        # Get genetic features
        genetic_features = torch.FloatTensor(self.genetic_data[idx])
        
        # Get MRI image
        image_data = self.mri_data.iloc[idx]['image']
        
        # Handle different image formats
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            raise ValueError(f"Unexpected image data format at index {idx}")
        
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        # Get label if available
        if self.labels is not None:
            label = torch.LongTensor([self.labels[idx]])[0]
            return genetic_features, image, label
        
        return genetic_features, image
    
    @staticmethod
    def load_data(genetic_path, mri_path, labels_path=None):
        """
        Load data from files.
        
        Args:
            genetic_path: Path to preprocessed genetic data (.npz)
            mri_path: Path to MRI parquet file
            labels_path: Optional path to labels
            
        Returns:
            Tuple of (genetic_data, mri_data, labels)
        """
        # Load genetic data
        genetic_data = np.load(genetic_path)
        genetic_train = genetic_data['X_train']
        genetic_test = genetic_data['X_test']
        
        # Load MRI data
        mri_df = pd.read_parquet(mri_path)
        
        # Split MRI data to match genetic data splits
        # Assuming MRI data needs to be split similarly
        # If MRI data is smaller, use what we have
        split_idx = min(len(genetic_train), len(mri_df))
        mri_train = mri_df.iloc[:split_idx]
        
        # For test set, take next portion or use remaining data
        remaining = len(mri_df) - split_idx
        test_size = min(len(genetic_test), remaining)
        mri_test = mri_df.iloc[split_idx:split_idx + test_size] if remaining > 0 else mri_train.iloc[:len(genetic_test)]
        
        # Load labels if provided
        labels_train = None
        labels_test = None
        if labels_path:
            labels_data = np.load(labels_path)
            labels_train = labels_data['y_train']
            labels_test = labels_data['y_test']
            # Convert one-hot to class indices
            labels_train = np.argmax(labels_train, axis=1)
            labels_test = np.argmax(labels_test, axis=1)
        elif 'label' in mri_df.columns:
            # Use labels from MRI data
            labels_train = mri_train['label'].values
            labels_test = mri_test['label'].values
        
        return (genetic_train, genetic_test), (mri_train, mri_test), (labels_train, labels_test)


def create_dataloaders(genetic_train, genetic_test, mri_train, mri_test,
                      labels_train=None, labels_test=None,
                      batch_size=32, num_workers=0, shuffle_train=True):
    """
    Create DataLoader objects for training and testing.
    
    Args:
        genetic_train: Training genetic data
        genetic_test: Test genetic data
        mri_train: Training MRI data DataFrame
        mri_test: Test MRI data DataFrame
        labels_train: Training labels
        labels_test: Test labels
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    from torchvision import transforms
    
    # Define image transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MultiModalDataset(
        genetic_train, mri_train, labels_train, transform=train_transform
    )
    
    test_dataset = MultiModalDataset(
        genetic_test, mri_test, labels_test, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader

