"""
Genetic Variant Encoder
Processes genetic variant features using fully connected layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneticEncoder(nn.Module):
    """
    Encoder for genetic variant data.
    Takes 130-dimensional genetic features and encodes them into a latent representation.
    """
    
    def __init__(self, input_dim=130, hidden_dims=[256, 128, 64], dropout=0.3, output_dim=64):
        """
        Args:
            input_dim: Number of genetic features (default: 130)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            output_dim: Dimension of the output embedding
        """
        super(GeneticEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output projection layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the genetic encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        return self.encoder(x)

