import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

for directory in [PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

MODEL_CONFIG = {
    'latent_dim': 20,
    'num_epochs': 100,
    'batch_size': 1024,
    'learning_rate': 0.001,
    'regularization': 0.01,
    'dropout': 0.2,
}

DATA_CONFIG = {
    'min_reviews_per_user': 5,
    'min_reviews_per_business': 5,
    'max_users': None,
    'max_businesses': None,
    'train_test_split': 0.8,
    'min_rating': 1.0,
    'max_rating': 5.0,
}

EVAL_CONFIG = {
    'k_values': [5, 10, 20],
    'rating_threshold': 3.5,
}

RANDOM_SEED = 42
