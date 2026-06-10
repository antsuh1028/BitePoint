"""
Generate evaluation plots for the report.
Produces figures in results/figures/ for both MF and Bayesian VI models.
"""

import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from config import PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR

FIGURES_DIR     = Path(RESULTS_DIR) / 'figures'
MF_FIG_DIR      = FIGURES_DIR / 'mf'
BAY_FIG_DIR     = FIGURES_DIR / 'bayesian'
CMP_FIG_DIR     = FIGURES_DIR / 'comparison'
for d in [MF_FIG_DIR, BAY_FIG_DIR, CMP_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
MF_CKPT_DIR  = Path(MODELS_DIR) / 'mf_checkpoints'
BAY_CKPT_DIR = Path(MODELS_DIR) / 'bayesian_checkpoints'

plt.rcParams.update({
    'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.4,
    'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
})

MF_EPOCHS  = [10, 20, 30, 40, 50]
BAY_EPOCHS = [5, 10, 15, 20]

# Consistent model colours used across every figure
MF_COLOR  = 'steelblue'
BAY_COLOR = 'darkorange'
MF_LABEL  = 'Matrix Factorization (MF)'
BAY_LABEL = 'Bayesian VI'


# ── data helpers ─────────────────────────────────────────────────────────────

def load_data():
    train_df = pd.read_csv(Path(PROCESSED_DATA_DIR) / 'train.csv')
    test_df  = pd.read_csv(Path(PROCESSED_DATA_DIR) / 'test.csv')
    return train_df, test_df


def mf_predict(ckpt_path, user_idx, business_idx):
    c = np.load(ckpt_path)
    pred = (c['user_factors'][user_idx] * c['business_factors'][business_idx]).sum(1)
    pred += c['user_bias'][user_idx] + c['business_bias'][business_idx] + float(c['global_bias'][0])
    return np.clip(pred, 1, 5)


def bay_predict(ckpt_path, user_idx, business_idx):
    import torch
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)['state_dict']
    u_loc  = sd['user_loc'].numpy()
    b_loc  = sd['business_loc'].numpy()
    u_bias = sd['user_bias'].numpy()
    b_bias = sd['business_bias'].numpy()
    g_bias = sd['global_bias'].numpy()
    pred = (u_loc[user_idx] * b_loc[business_idx]).sum(1)
    pred += u_bias[user_idx] + b_bias[business_idx] + float(g_bias)
    return np.clip(pred, 1, 5)


def bay_uncertainty(ckpt_path, user_idx):
    """Return per-user posterior std (mean across latent dims)."""
    import torch
    import torch.nn.functional as F
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)['state_dict']
    u_scale = sd['user_scale'].numpy()
    # softplus to get actual std
    std = np.log1p(np.exp(u_scale))   # softplus
    return std[user_idx].mean(axis=1)


# ── figure 1: training curves side by side (MF loss | Bayesian ELBO) ─────────

def plot_training_curves():
    import torch

    mf_losses = np.load(MF_CKPT_DIR / 'checkpoint_epoch0050.npz')['losses']
    mf_epochs = np.arange(1, len(mf_losses) + 1)

    bay_losses, bay_ep = [], []
    for ep in BAY_EPOCHS:
        path = BAY_CKPT_DIR / f'bayesian_epoch{ep:04d}.pt'
        if path.exists():
            ck = torch.load(path, map_location='cpu', weights_only=False)
            bay_losses.append(ck['losses'][-1])
            bay_ep.append(ep)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left — MF
    ax = axes[0]
    ax.plot(mf_epochs, mf_losses, linewidth=2, color=MF_COLOR, label=MF_LABEL)
    for ep in MF_EPOCHS:
        ax.axvline(ep, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE + L2 Regularization Loss')
    ax.set_title(f'{MF_LABEL}\nTraining Loss')
    ax.legend(fontsize=9)

    # Right — Bayesian
    ax = axes[1]
    if bay_losses:
        ax.plot(bay_ep, bay_losses, 'o-', linewidth=2, color=BAY_COLOR, label=BAY_LABEL)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('−ELBO (lower = better)')
        ax.set_title(f'{BAY_LABEL}\nTraining −ELBO')
        ax.legend(fontsize=9)

    fig.suptitle('Training Convergence by Model', fontsize=12)
    fig.tight_layout()
    _save(fig, 'training_curves.png')


# ── figure 3: MF vs Bayesian — model comparison bar chart ────────────────────

def plot_model_comparison():
    mf_path  = Path(MODELS_DIR) / 'mf_results.json'
    bay_path = Path(MODELS_DIR) / 'bayesian_results.json'
    if not mf_path.exists() or not bay_path.exists():
        print('Missing results JSON, skipping comparison plot.')
        return

    with open(mf_path)  as f: mf  = json.load(f)
    with open(bay_path) as f: bay = json.load(f)

    metrics = ['mae', 'rmse', 'correlation', 'classification_accuracy']
    labels  = ['MAE ↓', 'RMSE ↓', 'Correlation ↑', 'Class. Acc. ↑']

    mf_test  = [mf['test_metrics'][m]  for m in metrics]
    bay_test = [bay['test_metrics'][m] for m in metrics]

    x, w = np.arange(len(metrics)), 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w/2, mf_test,  w, label=f'{MF_LABEL} (50 ep)',  color=MF_COLOR,  alpha=0.85)
    b2 = ax.bar(x + w/2, bay_test, w, label=f'{BAY_LABEL} (20 ep)', color=BAY_COLOR, alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_title('MF vs Bayesian VI — Test Set Performance')
    ax.legend()
    ax.set_ylim(0, max(mf_test + bay_test) * 1.18)
    fig.tight_layout()
    _save(fig, 'model_comparison.png')


# ── figure 4: posterior uncertainty vs review count ──────────────────────────

def plot_uncertainty_vs_reviews(train_df, test_df):
    ckpt = BAY_CKPT_DIR / 'bayesian_epoch0020.pt'
    if not ckpt.exists():
        print('Bayesian checkpoint not found, skipping uncertainty plot.')
        return

    # Review count per user from training data
    review_counts = train_df.groupby('user_idx').size().reset_index(name='n_reviews')

    # Sample test users (avoid OOM on full 809K)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(test_df), size=min(20000, len(test_df)), replace=False)
    sample_df = test_df.iloc[sample_idx]

    u_idx = sample_df['user_idx'].values
    uncertainty = bay_uncertainty(ckpt, u_idx)

    merged = pd.DataFrame({'user_idx': u_idx, 'uncertainty': uncertainty})
    merged = merged.merge(review_counts, on='user_idx', how='left').dropna()

    # Bin by review count
    bins  = [0, 5, 10, 20, 50, 100, 500, 9999]
    labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '101-500', '500+']
    merged['bin'] = pd.cut(merged['n_reviews'], bins=bins, labels=labels)
    grouped = merged.groupby('bin', observed=True)['uncertainty'].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grouped.index.astype(str), grouped.values, color=BAY_COLOR, alpha=0.85, label=BAY_LABEL)
    ax.legend(fontsize=9)
    ax.set_xlabel('Number of Training Reviews per User')
    ax.set_ylabel('Mean Posterior Std (latent factor uncertainty)')
    ax.set_title('Bayesian VI — Posterior Uncertainty vs User Activity\n'
                 '(cold-start users are more uncertain)')
    fig.tight_layout()
    _save(fig, 'uncertainty_vs_reviews.png')


# ── figure 5: error distributions — MF vs Bayesian ───────────────────────────

def plot_error_comparison(y_true, mf_pred, bay_pred):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, pred, label, color in zip(
        axes,
        [mf_pred, bay_pred],
        [f'{MF_LABEL} (50 ep)', f'{BAY_LABEL} (20 ep)'],
        [MF_COLOR, BAY_COLOR],
    ):
        errors = pred - y_true
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        ax.hist(errors, bins=60, color=color, alpha=0.8, density=True, edgecolor='white', lw=0.3)
        ax.axvline(0,    color='black', linestyle='--', linewidth=1)
        ax.axvline( mae, color='red',   linestyle='--', linewidth=1.1, label=f'±MAE={mae:.3f}')
        ax.axvline(-mae, color='red',   linestyle='--', linewidth=1.1)
        ax.set_xlabel('Prediction Error')
        ax.set_title(f'{label}\nRMSE={rmse:.3f}')
        ax.legend(fontsize=9)

    axes[0].set_ylabel('Density')
    fig.suptitle('Test Set Error Distribution', fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, 'error_comparison.png')


# ── figure 6: actual vs predicted distributions (Bayesian) ───────────────────

def plot_rating_distributions(y_true, mf_pred, bay_pred):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0.5, 5.5, 11)

    for ax, pred, label, color in zip(
        axes,
        [mf_pred, bay_pred],
        [f'{MF_LABEL} (50 ep)', f'{BAY_LABEL} (20 ep)'],
        [MF_COLOR, BAY_COLOR],
    ):
        ax.hist(y_true, bins=bins, alpha=0.55, label='Actual',    color='gray',  density=True)
        ax.hist(pred,   bins=bins, alpha=0.7,  label='Predicted', color=color,   density=True)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend(fontsize=9)

    fig.suptitle('Actual vs Predicted Rating Distributions', fontsize=12)
    fig.tight_layout()
    _save(fig, 'rating_distributions.png')


# ── helper ────────────────────────────────────────────────────────────────────

_SUBFOLDER_MAP = {
    'training_curves':       MF_FIG_DIR,   # combined → mf/
    'loss_curve':            MF_FIG_DIR,
    'checkpoint_metrics':    MF_FIG_DIR,
    'error_distribution':    MF_FIG_DIR,
    'train_test_comparison': MF_FIG_DIR,
    'bayesian_elbo_curve':   BAY_FIG_DIR,
    'uncertainty_vs_reviews': BAY_FIG_DIR,
    'model_comparison':      CMP_FIG_DIR,
    'error_comparison':      CMP_FIG_DIR,
    'rating_distributions':  CMP_FIG_DIR,
}

def _save(fig, name):
    stem = name.replace('.png', '')
    folder = _SUBFOLDER_MAP.get(stem, FIGURES_DIR)
    out = folder / name
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data...')
    train_df, test_df = load_data()

    user_idx     = test_df['user_idx'].values
    business_idx = test_df['business_idx'].values
    y_true       = test_df['stars'].values.astype(np.float32)

    print('Generating MF predictions...')
    mf_pred = mf_predict(MF_CKPT_DIR / 'checkpoint_epoch0050.npz', user_idx, business_idx)

    print('Generating Bayesian predictions...')
    bay_pred = bay_predict(BAY_CKPT_DIR / 'bayesian_epoch0020.pt', user_idx, business_idx)

    print('\nGenerating figures...')
    plot_training_curves()
    plot_model_comparison()
    plot_uncertainty_vs_reviews(train_df, test_df)
    plot_error_comparison(y_true, mf_pred, bay_pred)
    plot_rating_distributions(y_true, mf_pred, bay_pred)

    print(f'\nAll figures saved to: {FIGURES_DIR}')
    print('\n=== Test Metrics ===')
    for name, pred in [('MF', mf_pred), ('Bayesian', bay_pred)]:
        mae  = np.mean(np.abs(pred - y_true))
        rmse = np.sqrt(np.mean((pred - y_true)**2))
        corr = np.corrcoef(y_true, pred)[0, 1]
        print(f'  {name:10s}  MAE={mae:.4f}  RMSE={rmse:.4f}  Corr={corr:.4f}')
