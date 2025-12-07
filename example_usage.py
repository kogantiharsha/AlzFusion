import sys
import os
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import MultiModalFusionModel
from src.data import create_dataloaders
from src.training import Trainer
from src.utils import Config


def example_training():
    print("="*60)
    print("EXAMPLE: Training the Multi-Modal Model")
    print("="*60)
    
    print("\n1. Loading data...")
    genetic_data = np.load(Config.GENETIC_DATA_PATH)
    genetic_train = genetic_data['X_train']
    genetic_test = genetic_data['X_test']
    
    mri_train_df = pd.read_parquet(Config.MRI_TRAIN_PATH)
    mri_test_df = pd.read_parquet(Config.MRI_TEST_PATH)
    
    min_train = min(len(genetic_train), len(mri_train_df))
    min_test = min(len(genetic_test), len(mri_test_df))
    
    genetic_train = genetic_train[:min_train]
    genetic_test = genetic_test[:min_test]
    mri_train_df = mri_train_df.iloc[:min_train]
    mri_test_df = mri_test_df.iloc[:min_test]
    
    labels_train = mri_train_df['label'].values if 'label' in mri_train_df.columns else None
    labels_test = mri_test_df['label'].values if 'label' in mri_test_df.columns else None
    
    print(f"   Training samples: {len(genetic_train)}")
    print(f"   Test samples: {len(genetic_test)}")
    
    print("\n2. Creating data loaders...")
    train_loader, test_loader = create_dataloaders(
        genetic_train, genetic_test,
        mri_train_df, mri_test_df,
        labels_train, labels_test,
        batch_size=Config.BATCH_SIZE
    )
    
    print("\n3. Creating model...")
    model = MultiModalFusionModel(
        genetic_input_dim=Config.GENETIC_INPUT_DIM,
        genetic_hidden_dims=Config.GENETIC_HIDDEN_DIMS,
        mri_output_dim=Config.MRI_OUTPUT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
        use_attention=Config.USE_ATTENTION
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n4. Setting up trainer...")
    device = torch.device(Config.DEVICE)
    trainer = Trainer(
        model=model,
        device=device,
        num_classes=Config.NUM_CLASSES,
        lr=Config.LEARNING_RATE
    )
    
    print("\n5. Training model (5 epochs for demonstration)...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=5,
        save_dir=Config.MODEL_SAVE_DIR,
        log_dir=Config.LOG_DIR
    )
    
    print("\n✓ Training example completed!")
    return model, history


def example_inference(model_path=None):
    print("\n" + "="*60)
    print("EXAMPLE: Inference with the Multi-Modal Model")
    print("="*60)
    
    if model_path is None:
        model_path = os.path.join(Config.MODEL_SAVE_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Model not found at {model_path}")
        return
    
    print(f"\n1. Loading model from {model_path}...")
    device = torch.device(Config.DEVICE)
    
    model = MultiModalFusionModel(
        genetic_input_dim=Config.GENETIC_INPUT_DIM,
        genetic_hidden_dims=Config.GENETIC_HIDDEN_DIMS,
        mri_output_dim=Config.MRI_OUTPUT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
        use_attention=Config.USE_ATTENTION
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("   ✓ Model loaded successfully")
    
    print("\n2. Preparing example data...")
    genetic_features = np.random.randn(1, Config.GENETIC_INPUT_DIM).astype(np.float32)
    genetic_tensor = torch.FloatTensor(genetic_features).to(device)
    
    mri_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    print("   ✓ Data prepared")
    
    print("\n3. Running inference...")
    with torch.no_grad():
        logits, attention_weights = model(genetic_tensor, mri_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
    
    class_names = ['Non-AD', 'MCI', 'AD', 'Other', 'Expression', 
                   'Fluid biomarker', 'Imaging', 'Neuropathology', 'Cognitive']
    
    print(f"\n   Predicted Class: {class_names[predicted_class]} (Class {predicted_class})")
    print(f"   Confidence: {probs[0][predicted_class].item()*100:.2f}%")
    
    print("\n✓ Inference example completed!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Example usage of the Multi-Modal Alzheimer\'s Prediction System')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'both'], 
                       default='both', help='Mode to run')
    parser.add_argument('--model_path', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        model, history = example_training()
    
    if args.mode in ['inference', 'both']:
        example_inference(args.model_path)
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
