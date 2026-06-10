"""
Data loading and preprocessing for Yelp dataset.

The Yelp dataset can be downloaded from: https://www.yelp.com/dataset
Place the JSON files in data/raw/
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_CONFIG, RANDOM_SEED
)

def load_yelp_json(filename: str) -> pd.DataFrame:
    """
    Load a Yelp JSON file into a pandas DataFrame.
    
    Args:
        filename: Name of the JSON file in RAW_DATA_DIR
        
    Returns:
        DataFrame with the data
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    return pd.DataFrame(data)


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
    
    # Filter by rating range
    reviews_df = reviews_df[
        (reviews_df['stars'] >= DATA_CONFIG['min_rating']) &
        (reviews_df['stars'] <= DATA_CONFIG['max_rating'])
    ]
    
    # Remove duplicates
    reviews_df = reviews_df.drop_duplicates(
        subset=['user_id', 'business_id'], keep='first'
    )
    
    return reviews_df


def filter_data(
    reviews_df: pd.DataFrame,
    users_df: pd.DataFrame = None,
    businesses_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter data based on minimum review counts and limits.
    
    Args:
        reviews_df: Review data
        users_df: User data (optional)
        businesses_df: Business data (optional)
        
    Returns:
        Tuple of (filtered_reviews, filtered_users, filtered_businesses)
    """
    # Filter users by minimum review count
    user_counts = reviews_df['user_id'].value_counts()
    valid_users = user_counts[
        user_counts >= DATA_CONFIG['min_reviews_per_user']
    ].index
    reviews_df = reviews_df[reviews_df['user_id'].isin(valid_users)]
    
    # Filter businesses by minimum review count
    business_counts = reviews_df['business_id'].value_counts()
    valid_businesses = business_counts[
        business_counts >= DATA_CONFIG['min_reviews_per_business']
    ].index
    reviews_df = reviews_df[reviews_df['business_id'].isin(valid_businesses)]
    
    # Apply limits if specified
    if DATA_CONFIG['max_users']:
        valid_users = valid_users[:DATA_CONFIG['max_users']]
        reviews_df = reviews_df[reviews_df['user_id'].isin(valid_users)]
    
    if DATA_CONFIG['max_businesses']:
        valid_businesses = valid_businesses[:DATA_CONFIG['max_businesses']]
        reviews_df = reviews_df[reviews_df['business_id'].isin(valid_businesses)]
    
    # Filter user and business dataframes if provided
    if users_df is not None:
        users_df = users_df[users_df['user_id'].isin(reviews_df['user_id'])]
    
    if businesses_df is not None:
        businesses_df = businesses_df[
            businesses_df['business_id'].isin(reviews_df['business_id'])
        ]
    
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
    Split data into train and test sets.
    
    Args:
        reviews_df: Review data
        train_ratio: Fraction to use for training
        seed: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    np.random.seed(seed)
    
    # Shuffle and split
    shuffled = reviews_df.sample(frac=1.0, random_state=seed)
    split_idx = int(len(shuffled) * train_ratio)
    
    train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
    test_df = shuffled.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, test_df


def load_processed_data():
    """
    Load already-processed train/test CSVs and mappings if they exist.

    Returns:
        Dictionary with processed data and metadata, or None if files are missing.
    """
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
    mappings_path = os.path.join(PROCESSED_DATA_DIR, 'mappings.pkl')

    if not all(os.path.exists(p) for p in [train_path, test_path, mappings_path]):
        return None

    print("Loading processed data from disk...")
    import pickle
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    return {
        'train': train_df,
        'test': test_df,
        'users': None,
        'businesses': None,
        'mappings': mappings,
    }


def load_and_prepare_data():
    """
    Main function to load and prepare all data.
    
    Returns:
        Dictionary with processed data and metadata
    """
    print("Loading Yelp data...")
    
    try:
        reviews_df = load_yelp_json('review.json')
        print(f"Loaded {len(reviews_df)} reviews")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the Yelp dataset from https://www.yelp.com/dataset")
        return None
    
    try:
        users_df = load_yelp_json('user.json')
        print(f"Loaded {len(users_df)} users")
    except:
        print("Warning: Could not load user data")
        users_df = None
    
    try:
        businesses_df = load_yelp_json('business.json')
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
