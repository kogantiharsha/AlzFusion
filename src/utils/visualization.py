"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_attention(attention_weights, genetic_features=None, mri_features=None, save_path=None):
    """
    Visualize attention weights.
    
    Args:
        attention_weights: Attention weight matrix
        genetic_features: Optional genetic feature names
        mri_features: Optional MRI feature names
        save_path: Optional path to save figure
    """
    if torch.is_tensor(attention_weights):
        attention_weights = attention_weights.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='YlOrRd', annot=True, fmt='.3f')
    plt.title('Cross-Modal Attention Weights')
    plt.xlabel('MRI Features')
    plt.ylabel('Genetic Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

