import torch
import torch.nn as nn
import torch.nn.functional as F
from .genetic_encoder import GeneticEncoder
from .mri_encoder import MRIEncoder


class CrossModalAttention(nn.Module):
    
    def __init__(self, dim=64):
        super(CrossModalAttention, self).__init__()
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, genetic_features, mri_features):
        q = self.query(genetic_features)
        k = self.key(mri_features)
        v = self.value(mri_features)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attended_mri = torch.matmul(attention_weights, v)
        
        q_reverse = self.query(mri_features)
        k_reverse = self.key(genetic_features)
        v_reverse = self.value(genetic_features)
        
        scores_reverse = torch.matmul(q_reverse, k_reverse.transpose(-2, -1)) * self.scale
        attention_weights_reverse = F.softmax(scores_reverse, dim=-1)
        attended_genetic = torch.matmul(attention_weights_reverse, v_reverse)
        
        return attended_genetic, attended_mri, attention_weights


class MultiModalFusionModel(nn.Module):
    
    def __init__(self, 
                 genetic_input_dim=130,
                 genetic_hidden_dims=[256, 128, 64],
                 mri_output_dim=64,
                 fusion_dim=128,
                 num_classes=9,
                 dropout=0.3,
                 use_attention=True):
        super(MultiModalFusionModel, self).__init__()
        
        self.use_attention = use_attention
        
        self.genetic_encoder = GeneticEncoder(
            input_dim=genetic_input_dim,
            hidden_dims=genetic_hidden_dims,
            dropout=dropout,
            output_dim=64
        )
        
        self.mri_encoder = MRIEncoder(
            output_dim=mri_output_dim,
            pretrained=True
        )
        
        if use_attention:
            self.attention = CrossModalAttention(dim=64)
            self.fusion = nn.Sequential(
                nn.Linear(64 * 2, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(64 * 2, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, genetic_features, mri_features):
        genetic_emb = self.genetic_encoder(genetic_features)
        mri_emb = self.mri_encoder(mri_features)
        
        if self.use_attention:
            attended_genetic, attended_mri, attention_weights = self.attention(
                genetic_emb, mri_emb
            )
            fused = torch.cat([attended_genetic, attended_mri], dim=1)
        else:
            fused = torch.cat([genetic_emb, mri_emb], dim=1)
            attention_weights = None
        
        fused_features = self.fusion(fused)
        logits = self.classifier(fused_features)
        
        if self.use_attention:
            return logits, attention_weights
        return logits, None
