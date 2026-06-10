import os
import numpy as np
import pandas as pd
import pickle

from data_loader import load_and_prepare_data, load_processed_data
from model import SimpleMatrixFactorization, LatentFactorModel, VariationalInference
from evaluation import evaluate_model
from config import MODELS_DIR

MODEL            = 'bayesian' #mf or bayesin
EPOCHS           = 20
BATCH_SIZE       = 4096
LATENT_DIM       = 20
LEARNING_RATE    = 0.01
REGULARIZATION   = 0.01


def main():
    print("YELP RESTAURANT RECOMMENDER - PROBABILISTIC GRAPHICAL MODEL\n")

    data = load_processed_data() or load_and_prepare_data()

    if data is None:
        print("Failed to load data. Exiting.")
        return

    train_df   = data['train']
    test_df    = data['test']
    mappings   = data['mappings']
    num_users  = mappings['num_users']
    num_businesses = mappings['num_businesses']

    print(f"Users: {num_users}, Businesses: {num_businesses}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    train_user_idx     = train_df['user_idx'].values
    train_business_idx = train_df['business_idx'].values
    train_ratings      = train_df['stars'].values.astype(np.float32)

    test_user_idx     = test_df['user_idx'].values
    test_business_idx = test_df['business_idx'].values
    test_ratings      = test_df['stars'].values.astype(np.float32)


    if MODEL == 'mf':
        model = SimpleMatrixFactorization(
            num_users=num_users,
            num_businesses=num_businesses,
            latent_dim=LATENT_DIM,
            learning_rate=LEARNING_RATE,
            regularization=REGULARIZATION
        )
        model.train(
            train_user_idx, train_business_idx, train_ratings,
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )

        train_pred = model.predict(train_user_idx, train_business_idx)
        test_pred  = model.predict(test_user_idx, test_business_idx)

    elif MODEL == 'bayesian':
        import torch

        torch_model = LatentFactorModel(
            num_users=num_users,
            num_businesses=num_businesses,
            latent_dim=LATENT_DIM,
            global_bias=float(train_ratings.mean()),
        )

        vi = VariationalInference(torch_model, learning_rate=LEARNING_RATE, num_epochs=EPOCHS)
        vi.train(
            train_user_idx, train_business_idx, train_ratings,
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )

        train_pred = vi.predict(torch.from_numpy(train_user_idx).long(), torch.from_numpy(train_business_idx).long())
        test_pred  = vi.predict(torch.from_numpy(test_user_idx).long(),  torch.from_numpy(test_business_idx).long())

    else:
        raise ValueError(f"Unknown model: {MODEL}")

    print("\nTraining evaluation:")
    train_results = evaluate_model(train_ratings, train_pred)

    print("\nTesting evaluations:")
    test_results = evaluate_model(test_ratings, test_pred)

    import json
    config = {'model': MODEL, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
              'latent_dim': LATENT_DIM, 'learning_rate': LEARNING_RATE, 'regularization': REGULARIZATION}
    with open(os.path.join(MODELS_DIR, f'{MODEL}_results.json'), 'w') as f:
        json.dump({'train_metrics': train_results, 'test_metrics': test_results, 'model_config': config}, f, indent=2)
    print("All done")
if __name__ == '__main__':
    main()
