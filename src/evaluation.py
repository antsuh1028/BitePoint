import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import EVAL_CONFIG


def evaluate_model(y_true, y_pred):
    # MAE = (1/n) * sum(|y_i - y_hat_i|)
    mae = mean_absolute_error(y_true, y_pred)
    # RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    threshold = EVAL_CONFIG['rating_threshold']
    acc = np.mean((y_true >= threshold) == (y_pred >= threshold))

    results = {'mae': mae, 'rmse': rmse, 'correlation': corr, 'classification_accuracy': acc}

    sorted_idx = np.argsort(y_pred)[::-1]
    total_relevant = np.sum(y_true >= threshold)
    for k in EVAL_CONFIG['k_values']:
        relevant = np.sum(y_true[sorted_idx[:k]] >= threshold)
        results[f'precision@{k}'] = relevant / k
        results[f'recall@{k}'] = relevant / total_relevant if total_relevant > 0 else 0.0

    print("Results:")
    print(f"  MAE:                {mae:.4f}")
    print(f"  RMSE:               {rmse:.4f}")
    print(f"  Correlation:        {corr:.4f}")
    print(f"  Classification Acc: {acc:.4f}")

    return results
