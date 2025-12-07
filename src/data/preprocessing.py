"""
Data preprocessing utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_genetic_data(data_path, normalize=True):
    """
    Load and preprocess genetic variant data.
    
    Args:
        data_path: Path to preprocessed .npz file
        normalize: Whether to normalize features
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    data = np.load(data_path)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler if normalize else None


def preprocess_mri_data(data_path):
    """
    Load MRI data from parquet file.
    
    Args:
        data_path: Path to parquet file
        
    Returns:
        DataFrame with image and label columns
    """
    df = pd.read_parquet(data_path)
    return df

