import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models import MultiModalFusionModel
from src.data import create_dataloaders
from src.training import Metrics
from src.utils import Config
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Modal Alzheimer\'s Prediction Model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--genetic_data', type=str, default=Config.GENETIC_DATA_PATH)
    parser.add_argument('--mri_test', type=str, default=Config.MRI_TEST_PATH)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--num_classes', type=int, default=Config.NUM_CLASSES)
    parser.add_argument('--save_results', type=str, default=None)
    
    args = parser.parse_args()
    
    device = torch.device(Config.DEVICE)
    print(f'Using device: {device}')
    
    print('\nLoading test data...')
    genetic_data = np.load(args.genetic_data)
    genetic_test = genetic_data['X_test']
    
    mri_test_df = pd.read_parquet(args.mri_test)
    
    min_test_size = min(len(genetic_test), len(mri_test_df))
    genetic_test = genetic_test[:min_test_size]
    mri_test_df = mri_test_df.iloc[:min_test_size]
    
    labels_test = mri_test_df['label'].values if 'label' in mri_test_df.columns else None
    
    print(f'Test samples: {len(genetic_test)}')
    
    _, test_loader = create_dataloaders(
        genetic_test, genetic_test,
        mri_test_df, mri_test_df,
        labels_test, labels_test,
        batch_size=args.batch_size,
        num_workers=Config.NUM_WORKERS,
        shuffle_train=False
    )
    
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
    
    print(f'Loaded model from epoch {checkpoint.get("epoch", "unknown")}')
    print(f'Model Val Acc: {checkpoint.get("val_acc", "unknown"):.2f}%')
    
    print('\nEvaluating model...')
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for genetic_features, mri_features, labels in test_loader:
            genetic_features = genetic_features.to(device)
            mri_features = mri_features.to(device)
            labels = labels.to(device)
            
            logits, _ = model(genetic_features, mri_features)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = Metrics(num_classes=args.num_classes)
    results = metrics.compute_metrics(all_labels, all_preds, all_probs)
    
    print('\n' + '='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    print(f'Accuracy: {results["accuracy"]:.4f}')
    print(f'Precision (Macro): {results["precision_macro"]:.4f}')
    print(f'Recall (Macro): {results["recall_macro"]:.4f}')
    print(f'F1-Score (Macro): {results["f1_macro"]:.4f}')
    print(f'F1-Score (Weighted): {results["f1_weighted"]:.4f}')
    print('\nPer-Class Metrics:')
    for i, class_name in enumerate(metrics.class_names):
        print(f'  {class_name}:')
        print(f'    Precision: {results["precision_per_class"][i]:.4f}')
        print(f'    Recall: {results["recall_per_class"][i]:.4f}')
        print(f'    F1: {results["f1_per_class"][i]:.4f}')
        print(f'    Support: {results["support"][i]}')
    
    print('\nGenerating confusion matrix...')
    os.makedirs('results', exist_ok=True)
    metrics.plot_confusion_matrix(
        results['confusion_matrix'],
        save_path='results/confusion_matrix.png'
    )
    
    print('\nClassification Report:')
    metrics.print_classification_report(all_labels, all_preds)
    
    if args.save_results:
        import json
        results_dict = {
            'accuracy': float(results['accuracy']),
            'precision_macro': float(results['precision_macro']),
            'recall_macro': float(results['recall_macro']),
            'f1_macro': float(results['f1_macro']),
            'f1_weighted': float(results['f1_weighted'])
        }
        with open(args.save_results, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f'\nResults saved to {args.save_results}')


if __name__ == '__main__':
    main()
