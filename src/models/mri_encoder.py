import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
import io


class MRIEncoder(nn.Module):
    
    def __init__(self, output_dim=64, pretrained=True):
        super(MRIEncoder, self).__init__()
        
        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = resnet.fc.in_features
        
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.projection(features)
    
    @staticmethod
    def preprocess_image(image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image)
