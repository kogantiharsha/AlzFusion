"""
Inference script for multi-modal Alzheimer's prediction model
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from PIL import Image
from src.models import MultiModalFusionModel
from src.models.mri_encoder import MRIEncoder
from src.utils import Config


def main():
    parser = argparse.ArgumentParser(description='Inference with Multi-Modal Alzheimer\'s Prediction Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--genetic_features', type=str, required=True,
                       help='Path to genetic features file (.npy) or comma-separated values')
    parser.add_argument('--mri_image', type=str, required=True,
                       help='Path to MRI image file')
    parser.add_argument('--num_classes', type=int, default=Config.NUM_CLASSES,
                       help='Number of classes')
    parser.add_argument('--class_names', type=str, nargs='+',
                       default=['Non-AD', 'MCI', 'AD', 'Other', 'Expression', 
                               'Fluid biomarker', 'Imaging', 'Neuropathology', 'Cognitive'],
                       help='Class names')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(Config.DEVICE)
    print(f'Using device: {device}')
    
    # Load model
    print('\nLoading model...')
    model = MultiModalFusionModel(
        genetic_input_dim=Config.GENETIC_INPUT_DIM,
        genetic_hidden_dims=Config.GENETIC_HIDDEN_DIMS,
        mri_output_dim=Config.MRI_OUTPUT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_classes=args.num_classes,
        dropout=Config.DROPOUT,
        use_attention=Config.USE_ATTENTION
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load genetic features
    print('\nLoading genetic features...')
    if args.genetic_features.endswith('.npy'):
        genetic_features = np.load(args.genetic_features)
    else:
        # Assume comma-separated values
        genetic_features = np.array([float(x) for x in args.genetic_features.split(',')])
    
    if len(genetic_features.shape) == 1:
        genetic_features = genetic_features.reshape(1, -1)
    
    if genetic_features.shape[1] != Config.GENETIC_INPUT_DIM:
        raise ValueError(f'Expected {Config.GENETIC_INPUT_DIM} genetic features, got {genetic_features.shape[1]}')
    
    genetic_tensor = torch.FloatTensor(genetic_features).to(device)
    
    # Load and preprocess MRI image
    print('Loading MRI image...')
    mri_image = Image.open(args.mri_image)
    if mri_image.mode != 'RGB':
        mri_image = mri_image.convert('RGB')
    
    # Preprocess image
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    mri_tensor = transform(mri_image).unsqueeze(0).to(device)
    
    # Run inference
    print('\nRunning inference...')
    with torch.no_grad():
        logits, attention_weights = model(genetic_tensor, mri_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # Display results
    print('\n' + '='*50)
    print('PREDICTION RESULTS')
    print('='*50)
    print(f'Predicted Class: {args.class_names[predicted_class]} (Class {predicted_class})')
    print(f'Confidence: {probs[0][predicted_class].item()*100:.2f}%')
    print('\nClass Probabilities:')
    for i, class_name in enumerate(args.class_names):
        prob = probs[0][i].item() * 100
        marker = ' <--' if i == predicted_class else ''
        print(f'  {class_name}: {prob:.2f}%{marker}')
    
    if attention_weights is not None:
        print('\nAttention mechanism was used in fusion.')
    
    print('\nInference completed!')


if __name__ == '__main__':
    main()

