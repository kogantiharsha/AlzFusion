import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_genetic_data(data_path, normalize=True):
    data = np.load(data_path)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler if normalize else None


def preprocess_mri_data(data_path):
    df = pd.read_parquet(data_path)
    return df
