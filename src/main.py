"""
Main entry point for training and evaluating the Yelp recommendation model.

Usage:
    python main.py --model mf --epochs 100 --batch_size 32
    python main.py --model bayesian --latent_dim 20
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

from data_loader import load_and_prepare_data, load_processed_data
from model import SimpleMatrixFactorization, LatentFactorModel, VariationalInference
from evaluation import evaluate_model, ablation_study
from config import MODEL_CONFIG, PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED


def create_data_loader(df, batch_size=32):
    """Create a simple batch iterator."""
    num_samples = len(df)
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        batch = df.iloc[i:end_idx]
        
        yield (
            batch['user_idx'].values,
            batch['business_idx'].values,
            batch['stars'].values.astype(np.float32)
        )


def main(args):
    """Main training and evaluation pipeline."""
    
    print("="*70)
    print("YELP RESTAURANT RECOMMENDER - PROBABILISTIC GRAPHICAL MODEL")
    print("="*70)
    
    # Load data
    print("\n[1] Loading and preparing data...")
    data = load_processed_data() or load_and_prepare_data()

    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    train_df = data['train']
    test_df = data['test']
    mappings = data['mappings']
    
    num_users = mappings['num_users']
    num_businesses = mappings['num_businesses']
    
    print(f"    Users: {num_users}, Businesses: {num_businesses}")
    print(f"    Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Get train/test data
    train_user_idx = train_df['user_idx'].values
    train_business_idx = train_df['business_idx'].values
    train_ratings = train_df['stars'].values.astype(np.float32)
    
    test_user_idx = test_df['user_idx'].values
    test_business_idx = test_df['business_idx'].values
    test_ratings = test_df['stars'].values.astype(np.float32)
    
    # Train model
    print(f"\n[2] Training {args.model} model...")
    
    if args.model == 'mf':
        # Matrix Factorization
        model = SimpleMatrixFactorization(
            num_users=num_users,
            num_businesses=num_businesses,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            regularization=args.regularization
        )
        
        model.train(
            train_user_idx, train_business_idx, train_ratings,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=str(Path(MODELS_DIR) / f'{args.model}_checkpoints'),
        )
        
        # Make predictions
        print("\n[3] Making predictions...")
        train_pred = model.predict(train_user_idx, train_business_idx)
        test_pred = model.predict(test_user_idx, test_business_idx)
        
    elif args.model == 'bayesian':
        # Bayesian Latent Factor Model (Pyro)
        print("    Note: Bayesian model requires more computational resources")
        print("          Consider using 'mf' model for faster experimentation")
        
        import torch
        
        torch_model = LatentFactorModel(
            num_users=num_users,
            num_businesses=num_businesses,
            latent_dim=args.latent_dim,
            global_bias=float(train_ratings.mean()),
        )
        
        vi = VariationalInference(
            torch_model,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs
        )

        # Train with mini-batch SVI
        vi.train(
            train_user_idx, train_business_idx, train_ratings,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=str(Path(MODELS_DIR) / f'{args.model}_checkpoints'),
        )

        # Make predictions
        print("\n[3] Making predictions...")
        test_user_idx_torch = torch.from_numpy(test_user_idx).long()
        test_business_idx_torch = torch.from_numpy(test_business_idx).long()
        train_user_idx_torch = torch.from_numpy(train_user_idx).long()
        train_business_idx_torch = torch.from_numpy(train_business_idx).long()

        train_pred = vi.predict(train_user_idx_torch, train_business_idx_torch)
        test_pred = vi.predict(test_user_idx_torch, test_business_idx_torch)
    
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Evaluate
    print("\n[4] Evaluating model...")
    print("\n--- Training Set Performance ---")
    train_results = evaluate_model(train_ratings, train_pred, verbose=True)
    
    print("\n--- Test Set Performance ---")
    test_results = evaluate_model(test_ratings, test_pred, verbose=True)
    
    # Save results
    print("\n[5] Saving results...")
    
    # Save predictions
    results_data = {
        'train_pred': train_pred,
        'test_pred': test_pred,
        'train_true': train_ratings,
        'test_true': test_ratings,
        'train_metrics': train_results,
        'test_metrics': test_results,
    }
    
    import json
    with open(Path(MODELS_DIR) / f'{args.model}_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            'train_metrics': train_results,
            'test_metrics': test_results,
            'model_config': vars(args),
        }
        json.dump(json_data, f, indent=2)
    
    print(f"    Results saved to {MODELS_DIR}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model:                {args.model}")
    print(f"Latent Dimension:     {args.latent_dim}")
    print(f"Epochs:               {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"\nTest Set Performance:")
    print(f"  MAE:                {test_results['mae']:.4f}")
    print(f"  RMSE:               {test_results['rmse']:.4f}")
    print(f"  Correlation:        {test_results['correlation']:.4f}")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate Yelp restaurant recommender model'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['mf', 'bayesian'],
        default='mf',
        help='Model type: mf (matrix factorization) or bayesian'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=MODEL_CONFIG['num_epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=MODEL_CONFIG['batch_size'],
        help='Batch size for training'
    )
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=MODEL_CONFIG['latent_dim'],
        help='Dimension of latent factors'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=MODEL_CONFIG['learning_rate'],
        help='Learning rate'
    )
    parser.add_argument(
        '--regularization',
        type=float,
        default=MODEL_CONFIG['regularization'],
        help='L2 regularization coefficient'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=10,
        help='Save a checkpoint and print progress report every N epochs (0 to disable)'
    )

    args = parser.parse_args()
    main(args)
