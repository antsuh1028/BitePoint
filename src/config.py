"""
Configuration parameters for the Yelp restaurant recommender model.
"""

import os

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model hyperparameters
MODEL_CONFIG = {
    'latent_dim': 20,           # Dimension of latent factors
    'num_epochs': 100,
    'batch_size': 1024,
    'learning_rate': 0.001,
    'regularization': 0.01,
    'dropout': 0.2,
}

# Data preprocessing parameters
DATA_CONFIG = {
    'min_reviews_per_user': 5,      # Filter users with fewer reviews
    'min_reviews_per_business': 5,  # Filter businesses with fewer reviews
    'max_users': None,              # Limit number of users (None = all)
    'max_businesses': None,         # Limit number of businesses (None = all)
    'train_test_split': 0.8,        # 80% train, 20% test
    'min_rating': 1.0,
    'max_rating': 5.0,
}

# Evaluation parameters
EVAL_CONFIG = {
    'k_values': [5, 10, 20],        # For ranking metrics
    'rating_threshold': 3.5,        # For classification metrics
}

# Random seed for reproducibility
RANDOM_SEED = 42
