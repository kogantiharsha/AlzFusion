"""
Evaluation metrics for the model
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class Metrics:
    """
    Class to compute and visualize evaluation metrics.
    """
    
    def __init__(self, num_classes=9, class_names=None):
        """
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
    
    def compute_metrics(self, y_true, y_pred, y_proba=None):
        """
        Compute comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy if tensors
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if y_proba is not None and torch.is_tensor(y_proba):
            y_proba = y_proba.cpu().numpy()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro = precision.mean()
        recall_macro = recall.mean()
        f1_macro = f1.mean()
        
        precision_weighted = np.average(precision, weights=support)
        recall_weighted = np.average(recall, weights=support)
        f1_weighted = np.average(f1, weights=support)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support': support,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path=None, figsize=(10, 8)):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Optional path to save figure
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_classification_report(self, y_true, y_pred):
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        print(classification_report(y_true, y_pred, target_names=self.class_names))

