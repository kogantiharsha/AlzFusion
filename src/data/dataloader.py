import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import io


class MultiModalDataset(Dataset):
    
    def __init__(self, genetic_data, mri_data, labels=None, transform=None):
        self.genetic_data = genetic_data
        self.mri_data = mri_data
        self.labels = labels
        self.transform = transform
        
        assert len(genetic_data) == len(mri_data), \
            f"Genetic data ({len(genetic_data)}) and MRI data ({len(mri_data)}) must have same length"
        
        if labels is not None:
            assert len(genetic_data) == len(labels), \
                f"Data ({len(genetic_data)}) and labels ({len(labels)}) must have same length"
    
    def __len__(self):
        return len(self.genetic_data)
    
    def __getitem__(self, idx):
        genetic_features = torch.FloatTensor(self.genetic_data[idx])
        image_data = self.mri_data.iloc[idx]['image']
        
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            raise ValueError(f"Unexpected image data format at index {idx}")
        
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        if self.labels is not None:
            label = torch.LongTensor([self.labels[idx]])[0]
            return genetic_features, image, label
        
        return genetic_features, image
    
    @staticmethod
    def load_data(genetic_path, mri_path, labels_path=None):
        genetic_data = np.load(genetic_path)
        genetic_train = genetic_data['X_train']
        genetic_test = genetic_data['X_test']
        
        mri_df = pd.read_parquet(mri_path)
        
        split_idx = min(len(genetic_train), len(mri_df))
        mri_train = mri_df.iloc[:split_idx]
        
        remaining = len(mri_df) - split_idx
        test_size = min(len(genetic_test), remaining)
        mri_test = mri_df.iloc[split_idx:split_idx + test_size] if remaining > 0 else mri_train.iloc[:len(genetic_test)]
        
        labels_train = None
        labels_test = None
        if labels_path:
            labels_data = np.load(labels_path)
            labels_train = labels_data['y_train']
            labels_test = labels_data['y_test']
            labels_train = np.argmax(labels_train, axis=1)
            labels_test = np.argmax(labels_test, axis=1)
        elif 'label' in mri_df.columns:
            labels_train = mri_train['label'].values
            labels_test = mri_test['label'].values
        
        return (genetic_train, genetic_test), (mri_train, mri_test), (labels_train, labels_test)


def create_dataloaders(genetic_train, genetic_test, mri_train, mri_test,
                      labels_train=None, labels_test=None,
                      batch_size=32, num_workers=0, shuffle_train=True):
    from torchvision import transforms
    
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
    
    train_dataset = MultiModalDataset(
        genetic_train, mri_train, labels_train, transform=train_transform
    )
    
    test_dataset = MultiModalDataset(
        genetic_test, mri_test, labels_test, transform=test_transform
    )
    
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
