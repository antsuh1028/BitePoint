"""
Evaluation metrics for the recommendation system.

Includes rating prediction metrics and ranking metrics.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict
from config import EVAL_CONFIG


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def rating_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between predicted and actual ratings."""
    return np.corrcoef(y_true, y_pred)[0, 1]


def classify_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 3.5
) -> float:
    """
    Accuracy of classifying whether rating is above/below threshold.
    (Binary classification: like/dislike)
    """
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    return np.mean(y_true_binary == y_pred_binary)


def ndcg_at_k(
    user_ratings: Dict[int, Dict[int, float]],
    user_predictions: Dict[int, np.ndarray],
    k: int = 10
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    
    Args:
        user_ratings: Dict[user_id] = Dict[business_id] = rating
        user_predictions: Dict[user_id] = array of predicted ratings
        k: Top-k for evaluation
        
    Returns:
        Mean NDCG@k across all users
    """
    ndcg_scores = []
    
    for user_id, predictions in user_predictions.items():
        if user_id not in user_ratings:
            continue
        
        user_actual = user_ratings[user_id]
        
        # Get top-k indices
        top_k_idx = np.argsort(predictions)[-k:][::-1]
        
        # Compute DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_idx):
            # This is a simplified version - in practice you'd need to track
            # which business ID corresponds to each prediction
            if idx in user_actual:
                relevance = user_actual[idx] / 5.0  # Normalize to [0, 1]
                dcg += relevance / np.log2(i + 2)  # +2 for 1-indexing
        
        # Compute ideal DCG
        ideal_ratings = sorted(user_actual.values(), reverse=True)[:k]
        ideal_dcg = sum(
            r / 5.0 / np.log2(i + 2)
            for i, r in enumerate(ideal_ratings)
        )
        
        if ideal_dcg > 0:
            ndcg = dcg / ideal_dcg
        else:
            ndcg = 0.0
        
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: list = None
) -> Dict[str, float]:
    """
    Compute ranking metrics at different k values.
    
    This assumes we're ranking items (businesses) for a user.
    """
    if k_values is None:
        k_values = EVAL_CONFIG['k_values']
    
    metrics = {}
    
    # Sort by prediction score
    sorted_idx = np.argsort(y_pred)[::-1]
    
    for k in k_values:
        top_k_idx = sorted_idx[:k]
        
        # Precision@k: fraction of top-k that are relevant (rating >= threshold)
        threshold = EVAL_CONFIG['rating_threshold']
        relevant = np.sum(y_true[top_k_idx] >= threshold)
        precision_k = relevant / k if k > 0 else 0.0
        metrics[f'precision@{k}'] = precision_k
        
        # Recall@k
        total_relevant = np.sum(y_true >= threshold)
        recall_k = relevant / total_relevant if total_relevant > 0 else 0.0
        metrics[f'recall@{k}'] = recall_k
    
    return metrics


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        y_true: Actual ratings
        y_pred: Predicted ratings
        verbose: Whether to print results
        
    Returns:
        Dictionary with all evaluation metrics
    """
    results = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'correlation': rating_correlation(y_true, y_pred),
        'classification_accuracy': classify_accuracy(y_true, y_pred),
    }
    
    # Add ranking metrics
    ranking_metrics_result = ranking_metrics(y_true, y_pred)
    results.update(ranking_metrics_result)
    
    if verbose:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"MAE:                   {results['mae']:.4f}")
        print(f"RMSE:                  {results['rmse']:.4f}")
        print(f"Correlation:           {results['correlation']:.4f}")
        print(f"Classification Acc:    {results['classification_accuracy']:.4f}")
        print("\nRanking Metrics:")
        for key, value in results.items():
            if '@' in key:
                print(f"  {key}: {value:.4f}")
        print("="*50)
    
    return results


def ablation_study(
    model_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    model_names: list = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance across multiple models/configurations.
    
    Args:
        model_predictions: Dict[model_name] = predictions array
        y_true: Ground truth ratings
        model_names: Names of models (if None, uses keys from model_predictions)
        
    Returns:
        Dict with evaluation results for each model
    """
    if model_names is None:
        model_names = list(model_predictions.keys())
    
    results = {}
    
    for name in model_names:
        if name not in model_predictions:
            continue
        
        y_pred = model_predictions[name]
        results[name] = evaluate_model(y_true, y_pred, verbose=False)
    
    # Print comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    if results:
        # Get all metric keys
        metrics = list(results[model_names[0]].keys())
        
        # Print header
        print(f"{'Model':<20}", end='')
        for metric in metrics:
            print(f"{metric:<12}", end='')
        print()
        
        # Print results
        for model_name in model_names:
            if model_name in results:
                print(f"{model_name:<20}", end='')
                for metric in metrics:
                    value = results[model_name][metric]
                    print(f"{value:<12.4f}", end='')
                print()
    
    print("="*70)
    
    return results
