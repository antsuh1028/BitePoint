"""
Data loading and preprocessing for Yelp dataset.

The Yelp dataset can be downloaded from: https://www.yelp.com/dataset
Place the JSON files in data/raw/
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_CONFIG, RANDOM_SEED
)

_COLUMNS_TO_LOAD = {
    'yelp_academic_dataset_review.json': ['user_id', 'business_id', 'stars'],
    'yelp_academic_dataset_user.json': ['user_id', 'review_count'],
    'yelp_academic_dataset_business.json': ['business_id', 'name', 'categories', 'stars'],
}

_CHUNK_SIZE = 100_000  # Number of rows per chunk


def load_yelp_json(filename: str) -> pd.DataFrame:
    """
    Load a Yelp JSON file into a pandas DataFrame using chunked reading
    to avoid out-of-memory errors on large files.

    Only the columns listed in _COLUMNS_TO_LOAD are retained (when the
    file type is known), so the in-memory footprint is dramatically
    reduced compared to loading all columns at once.

    Args:
        filename: Name of the JSON file in RAW_DATA_DIR

    Returns:
        DataFrame with the data
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    keep_cols = _COLUMNS_TO_LOAD.get(filename)  # None → keep all columns

    chunks = []
    for chunk in pd.read_json(filepath, lines=True, chunksize=_CHUNK_SIZE):
        if keep_cols is not None:
            # Only keep columns that actually exist in this chunk
            cols = [c for c in keep_cols if c in chunk.columns]
            chunk = chunk[cols]
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def preprocess_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess review data.

    Args:
        reviews_df: DataFrame with review data

    Returns:
        Cleaned review DataFrame
    """
    # Ensure required columns exist
    required_cols = ['user_id', 'business_id', 'stars']
    for col in required_cols:
        if col not in reviews_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Filter by rating range — use .copy() to avoid SettingWithCopyWarning
    # on downstream column assignments.
    reviews_df = reviews_df[
        (reviews_df['stars'] >= DATA_CONFIG['min_rating']) &
        (reviews_df['stars'] <= DATA_CONFIG['max_rating'])
    ].copy()

    # Remove duplicates
    reviews_df = reviews_df.drop_duplicates(
        subset=['user_id', 'business_id'], keep='first'
    ).copy()

    return reviews_df


def filter_data(
    reviews_df: pd.DataFrame,
    users_df: pd.DataFrame = None,
    businesses_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter data based on minimum review counts using an iterative k-core
    approach, then optionally cap the number of users/businesses.

    A single-pass filter is insufficient: removing businesses with too few
    reviews can reduce some users' review counts below the minimum (and
    vice-versa). The loop below alternates until the review set stabilises,
    guaranteeing that *both* constraints hold simultaneously in the final
    dataset.

    Args:
        reviews_df: Review data
        users_df: User data (optional)
        businesses_df: Business data (optional)

    Returns:
        Tuple of (filtered_reviews, filtered_users, filtered_businesses)
    """
    min_user_reviews = DATA_CONFIG['min_reviews_per_user']
    min_biz_reviews  = DATA_CONFIG['min_reviews_per_business']

    # --- Iterative k-core filtering ---
    iteration = 0
    while True:
        num_reviews_before = len(reviews_df)

        # Remove users below minimum review count
        user_counts = reviews_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_reviews].index
        reviews_df = reviews_df[reviews_df['user_id'].isin(valid_users)]

        # Remove businesses below minimum review count
        business_counts = reviews_df['business_id'].value_counts()
        valid_businesses = business_counts[
            business_counts >= min_biz_reviews
        ].index
        reviews_df = reviews_df[
            reviews_df['business_id'].isin(valid_businesses)
        ]

        iteration += 1
        if len(reviews_df) == num_reviews_before:
            # Constraints are simultaneously satisfied — we are done.
            break

    print(f"    k-core filtering converged in {iteration} iteration(s). "
          f"Remaining reviews: {len(reviews_df)}")

    # --- Optional hard caps ---
    if DATA_CONFIG['max_users']:
        # Re-compute after k-core so the cap is applied to the clean set
        top_users = (
            reviews_df['user_id']
            .value_counts()
            .head(DATA_CONFIG['max_users'])
            .index
        )
        reviews_df = reviews_df[
            reviews_df['user_id'].isin(top_users)
        ].copy()

    if DATA_CONFIG['max_businesses']:
        top_businesses = (
            reviews_df['business_id']
            .value_counts()
            .head(DATA_CONFIG['max_businesses'])
            .index
        )
        reviews_df = reviews_df[
            reviews_df['business_id'].isin(top_businesses)
        ].copy()

    # --- Propagate to auxiliary dataframes ---
    if users_df is not None:
        users_df = users_df[
            users_df['user_id'].isin(reviews_df['user_id'])
        ].copy()

    if businesses_df is not None:
        businesses_df = businesses_df[
            businesses_df['business_id'].isin(reviews_df['business_id'])
        ].copy()

    return reviews_df, users_df, businesses_df


def create_mappings(reviews_df: pd.DataFrame) -> Dict:
    """
    Create ID-to-index mappings for users and businesses.
    
    Args:
        reviews_df: Review data
        
    Returns:
        Dictionary with mappings
    """
    unique_users = reviews_df['user_id'].unique()
    unique_businesses = reviews_df['business_id'].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    business_to_idx = {bid: idx for idx, bid in enumerate(unique_businesses)}
    
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_business = {idx: bid for bid, idx in business_to_idx.items()}
    
    return {
        'user_to_idx': user_to_idx,
        'business_to_idx': business_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_business': idx_to_business,
        'num_users': len(unique_users),
        'num_businesses': len(unique_businesses),
    }


def split_train_test(
    reviews_df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets, ensuring that every user and
    business in the test set is also present in the training set.

    Args:
        reviews_df: Review data (must contain 'user_id' and 'business_id')
        train_ratio: Fraction of each user's reviews to use for training
        seed: Random seed

    Returns:
        Tuple of (train_df, test_df)
    """
    rng = np.random.default_rng(seed)

    train_parts = []
    test_parts  = []

    for _, user_reviews in reviews_df.groupby('user_id', sort=False):
        user_reviews = user_reviews.sample(
            frac=1.0, random_state=int(rng.integers(0, 2**31))
        )
        n_train = max(1, int(np.ceil(len(user_reviews) * train_ratio)))
        train_parts.append(user_reviews.iloc[:n_train])
        if n_train < len(user_reviews):
            test_parts.append(user_reviews.iloc[n_train:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts else pd.DataFrame(columns=reviews_df.columns)

    # Safety net: drop test rows whose business was never seen in training
    train_businesses = set(train_df['business_id'])
    test_df = test_df[
        test_df['business_id'].isin(train_businesses)
    ].reset_index(drop=True)

    return train_df, test_df


def load_and_prepare_data():
    """
    Main function to load and prepare all data.
    
    Returns:
        Dictionary with processed data and metadata
    """
    print("Loading Yelp data...")
    
    try:
        reviews_df = load_yelp_json('yelp_academic_dataset_review.json')
        print(f"Loaded {len(reviews_df)} reviews")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the Yelp dataset from https://www.yelp.com/dataset")
        return None
    
    try:
        users_df = load_yelp_json('yelp_academic_dataset_user.json')
        print(f"Loaded {len(users_df)} users")
    except:
        print("Warning: Could not load user data")
        users_df = None
    
    try:
        businesses_df = load_yelp_json('yelp_academic_dataset_business.json')
        print(f"Loaded {len(businesses_df)} businesses")
    except:
        print("Warning: Could not load business data")
        businesses_df = None
    
    # Preprocess
    print("Preprocessing reviews...")
    reviews_df = preprocess_reviews(reviews_df)
    
    print("Filtering data...")
    reviews_df, users_df, businesses_df = filter_data(
        reviews_df, users_df, businesses_df
    )
    print(f"After filtering: {len(reviews_df)} reviews")
    
    # Create mappings
    print("Creating ID mappings...")
    mappings = create_mappings(reviews_df)
    
    # Apply mappings
    reviews_df['user_idx'] = reviews_df['user_id'].map(mappings['user_to_idx'])
    reviews_df['business_idx'] = reviews_df['business_id'].map(
        mappings['business_to_idx']
    )
    
    # Split data
    print("Splitting train/test...")
    train_df, test_df = split_train_test(
        reviews_df,
        train_ratio=DATA_CONFIG['train_test_split'],
        seed=RANDOM_SEED
    )
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Save processed data
    print("Saving processed data...")
    train_df.to_csv(
        os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False
    )
    test_df.to_csv(
        os.path.join(PROCESSED_DATA_DIR, 'test.csv'), index=False
    )
    
    # Save mappings
    import pickle
    with open(os.path.join(PROCESSED_DATA_DIR, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
    
    return {
        'train': train_df,
        'test': test_df,
        'users': users_df,
        'businesses': businesses_df,
        'mappings': mappings,
    }


if __name__ == '__main__':
    load_and_prepare_data()
