import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneticEncoder(nn.Module):
    
    def __init__(self, input_dim=130, hidden_dims=[256, 128, 64], dropout=0.3, output_dim=64):
        super(GeneticEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)
