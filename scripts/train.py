import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models import MultiModalFusionModel
from src.data import MultiModalDataset, create_dataloaders
from src.training import Trainer
from src.utils import Config
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Modal Alzheimer\'s Prediction Model')
    parser.add_argument('--genetic_data', type=str, default=Config.GENETIC_DATA_PATH)
    parser.add_argument('--mri_train', type=str, default=Config.MRI_TRAIN_PATH)
    parser.add_argument('--mri_test', type=str, default=Config.MRI_TEST_PATH)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--save_dir', type=str, default=Config.MODEL_SAVE_DIR)
    parser.add_argument('--log_dir', type=str, default=Config.LOG_DIR)
    parser.add_argument('--num_classes', type=int, default=Config.NUM_CLASSES)
    
    args = parser.parse_args()
    
    device = torch.device(Config.DEVICE)
    print(f'Using device: {device}')
    
    print('\nLoading data...')
    genetic_data = np.load(args.genetic_data)
    genetic_train = genetic_data['X_train']
    genetic_test = genetic_data['X_test']
    
    mri_train_df = pd.read_parquet(args.mri_train)
    mri_test_df = pd.read_parquet(args.mri_test)
    
    min_train_size = min(len(genetic_train), len(mri_train_df))
    min_test_size = min(len(genetic_test), len(mri_test_df))
    
    genetic_train = genetic_train[:min_train_size]
    genetic_test = genetic_test[:min_test_size]
    mri_train_df = mri_train_df.iloc[:min_train_size]
    mri_test_df = mri_test_df.iloc[:min_test_size]
    
    print(f'Training samples: {len(genetic_train)}')
    print(f'Test samples: {len(genetic_test)}')
    
    labels_train = mri_train_df['label'].values if 'label' in mri_train_df.columns else None
    labels_test = mri_test_df['label'].values if 'label' in mri_test_df.columns else None
    
    print('\nCreating data loaders...')
    train_loader, test_loader = create_dataloaders(
        genetic_train, genetic_test,
        mri_train_df, mri_test_df,
        labels_train, labels_test,
        batch_size=args.batch_size,
        num_workers=Config.NUM_WORKERS
    )
    
    print('\nCreating model...')
    model = MultiModalFusionModel(
        genetic_input_dim=Config.GENETIC_INPUT_DIM,
        genetic_hidden_dims=Config.GENETIC_HIDDEN_DIMS,
        mri_output_dim=Config.MRI_OUTPUT_DIM,
        fusion_dim=Config.FUSION_DIM,
        num_classes=args.num_classes,
        dropout=Config.DROPOUT,
        use_attention=Config.USE_ATTENTION
    )
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    trainer = Trainer(
        model=model,
        device=device,
        num_classes=args.num_classes,
        lr=args.lr
    )
    
    print('\nStarting training...')
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    print('\nTraining completed!')
    
    from src.utils import plot_training_history
    os.makedirs(args.save_dir, exist_ok=True)
    plot_training_history(history, save_path=os.path.join(args.save_dir, 'training_history.png'))


if __name__ == '__main__':
    main()
