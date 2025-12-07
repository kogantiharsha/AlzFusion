"""
MRI Image Encoder
Processes MRI brain images using convolutional neural networks
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
import io


class MRIEncoder(nn.Module):
    """
    Encoder for MRI brain images.
    Uses a ResNet-based architecture for feature extraction.
    """
    
    def __init__(self, output_dim=64, pretrained=True):
        """
        Args:
            output_dim: Dimension of the output embedding
            pretrained: Whether to use pretrained weights
        """
        super(MRIEncoder, self).__init__()
        
        # Use ResNet18 as backbone (lightweight and effective)
        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get the feature dimension from ResNet
        feature_dim = resnet.fc.in_features
        
        # Project to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Forward pass through the MRI encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Encoded representation of shape (batch_size, output_dim)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Flatten spatial dimensions
        features = features.view(features.size(0), -1)
        
        # Project to output dimension
        return self.projection(features)
    
    @staticmethod
    def preprocess_image(image_bytes):
        """
        Preprocess image bytes for the encoder.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed tensor
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image)

