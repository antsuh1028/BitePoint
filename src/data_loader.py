import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_CONFIG, RANDOM_SEED


def load_yelp_json(filename):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_json(filepath, lines=True)


def preprocess_reviews(reviews_df):
    for col in ['user_id', 'business_id', 'stars']:
        if col not in reviews_df.columns:
            raise ValueError(f"Missing required column: {col}")
    reviews_df = reviews_df[
        (reviews_df['stars'] >= DATA_CONFIG['min_rating']) &
        (reviews_df['stars'] <= DATA_CONFIG['max_rating'])
    ].copy()
    return reviews_df.drop_duplicates(subset=['user_id', 'business_id'], keep='first').copy()

#filters, and makes sure that each review is valid
def filter_data(reviews_df, users_df=None, businesses_df=None):
    user_counts = reviews_df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= DATA_CONFIG['min_reviews_per_user']].index
    reviews_df = reviews_df[reviews_df['user_id'].isin(valid_users)]

    biz_counts = reviews_df['business_id'].value_counts()
    valid_biz = biz_counts[biz_counts >= DATA_CONFIG['min_reviews_per_business']].index
    reviews_df = reviews_df[reviews_df['business_id'].isin(valid_biz)]

    print(f"    After filtering: {len(reviews_df)} reviews")

    if users_df is not None:
        users_df = users_df[users_df['user_id'].isin(reviews_df['user_id'])].copy()
    if businesses_df is not None:
        businesses_df = businesses_df[businesses_df['business_id'].isin(reviews_df['business_id'])].copy()

    return reviews_df, users_df, businesses_df

#creates the mapping between the reviews to the users
def create_mappings(reviews_df):
    unique_users = reviews_df['user_id'].unique()
    unique_businesses = reviews_df['business_id'].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    business_to_idx = {bid: idx for idx, bid in enumerate(unique_businesses)}

    return {
        'user_to_idx': user_to_idx,
        'business_to_idx': business_to_idx,
        'idx_to_user': {idx: uid for uid, idx in user_to_idx.items()},
        'idx_to_business': {idx: bid for bid, idx in business_to_idx.items()},
        'num_users': len(unique_users),
        'num_businesses': len(unique_businesses),
    }


def split_train_test(reviews_df, train_ratio=0.8, seed=RANDOM_SEED):
    train_df, test_df = train_test_split(reviews_df, train_size=train_ratio, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_processed_data():
    train_path    = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    test_path     = os.path.join(PROCESSED_DATA_DIR, 'test.csv')
    mappings_path = os.path.join(PROCESSED_DATA_DIR, 'mappings.pkl')

    if not all(os.path.exists(p) for p in [train_path, test_path, mappings_path]):
        return None

    import pickle
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    return {'train': train_df, 'test': test_df, 'users': None, 'businesses': None, 'mappings': mappings}


def load_and_prepare_data():

    try:
        reviews_df = load_yelp_json('review.json')[['user_id', 'business_id', 'stars']]
        print(f"Loaded {len(reviews_df)} reviews")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    try:
        users_df = load_yelp_json('user.json')[['user_id', 'review_count']]
        print(f"Loaded {len(users_df)} users")
    except:
        users_df = None

    try:
        businesses_df = load_yelp_json('business.json')[['business_id', 'name', 'categories', 'stars']]
        print(f"Loaded {len(businesses_df)} businesses")
    except:
        businesses_df = None

    reviews_df = preprocess_reviews(reviews_df)
    reviews_df, users_df, businesses_df = filter_data(reviews_df, users_df, businesses_df)

    mappings = create_mappings(reviews_df)
    reviews_df['user_idx'] = reviews_df['user_id'].map(mappings['user_to_idx'])
    reviews_df['business_idx'] = reviews_df['business_id'].map(mappings['business_to_idx'])

    train_df, test_df = split_train_test(reviews_df, train_ratio=DATA_CONFIG['train_test_split'], seed=RANDOM_SEED)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test.csv'), index=False)

    import pickle
    with open(os.path.join(PROCESSED_DATA_DIR, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)

    return {'train': train_df, 'test': test_df, 'users': users_df, 'businesses': businesses_df, 'mappings': mappings}


if __name__ == '__main__':
    load_and_prepare_data()
