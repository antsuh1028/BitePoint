"""
Probabilistic Graphical Model for Restaurant Recommendation.

This module implements a latent factor model using Pyro for Bayesian inference.
The model assumes:
- Each user has a latent preference vector
- Each restaurant has a latent feature vector
- Ratings are generated from the dot product of these vectors plus noise
"""

import numpy as np
from config import MODEL_CONFIG, RANDOM_SEED

np.random.seed(RANDOM_SEED)

try:
    import torch
    import torch.nn as nn
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import Adam
    torch.manual_seed(RANDOM_SEED)
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_NNModule = torch.nn.Module if _TORCH_AVAILABLE else object


class LatentFactorModel(_NNModule):
    """
    Probabilistic latent factor model for rating prediction.

    Generative model:
        z_u ~ N(0, I)            user latent factors
        z_b ~ N(0, I)            business latent factors
        r   ~ N(z_u·z_b + b_u + b_b + mu, sigma^2)

    Variational posterior (mean-field diagonal Gaussian):
        q(z_u) = N(mu_u, diag(softplus(s_u)^2))
        q(z_b) = N(mu_b, diag(softplus(s_b)^2))
    """

    def __init__(self, num_users: int, num_businesses: int, latent_dim: int = 20,
                 global_bias: float = 3.75):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch and pyro-ppl are required for LatentFactorModel")
        super().__init__()
        self.num_users = num_users
        self.num_businesses = num_businesses
        self.latent_dim = latent_dim

        # Variational parameters for latent factors
        self.user_loc   = nn.Parameter(torch.randn(num_users, latent_dim) * 0.01)
        self.user_scale = nn.Parameter(torch.ones(num_users, latent_dim) * 0.5)

        self.business_loc   = nn.Parameter(torch.randn(num_businesses, latent_dim) * 0.01)
        self.business_scale = nn.Parameter(torch.ones(num_businesses, latent_dim) * 0.5)

        # Bias terms (point estimates — MAP)
        self.user_bias     = nn.Parameter(torch.zeros(num_users))
        self.business_bias = nn.Parameter(torch.zeros(num_businesses))
        self.global_bias   = nn.Parameter(torch.tensor(global_bias))
        self.log_noise     = nn.Parameter(torch.tensor(0.0))  # log(sigma)


class VariationalInference:
    """
    Mini-batch variational inference for LatentFactorModel.

    Uses the reparameterization trick with a closed-form KL divergence:
        ELBO = E_q[log p(r|z)] - KL(q(z_U)||p(z_U)) - KL(q(z_B)||p(z_B))

    The KL is computed analytically over ALL parameters each step (cheap — no
    sampling).  The likelihood is computed on a mini-batch and scaled up by
    N_total / batch_size so the two terms remain balanced throughout training.
    """

    def __init__(self, model: LatentFactorModel, learning_rate: float = 0.01,
                 num_epochs: int = 20):
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.losses = []
        self._optimizer = None  # created lazily so lr can be overridden

    # ── ELBO ─────────────────────────────────────────────────────────────────

    def _neg_elbo(self, u_idx, b_idx, ratings, n_total):
        F = nn.functional
        m = self.model
        B = len(ratings)

        # Unique users/businesses in this batch
        unique_u, u_inv = torch.unique(u_idx, return_inverse=True)
        unique_b, b_inv = torch.unique(b_idx, return_inverse=True)

        # Variational parameters for unique batch members only
        u_mu  = m.user_loc[unique_u]
        u_sig = F.softplus(m.user_scale[unique_u]) + 1e-6
        b_mu  = m.business_loc[unique_b]
        b_sig = F.softplus(m.business_scale[unique_b]) + 1e-6

        # Reparameterized samples: z = mu + sig * eps
        u_z = u_mu + u_sig * torch.randn_like(u_mu)
        b_z = b_mu + b_sig * torch.randn_like(b_mu)

        # Predicted rating (gather back to full batch size)
        pred = (u_z[u_inv] * b_z[b_inv]).sum(-1) \
             + m.user_bias[u_idx] + m.business_bias[b_idx] + m.global_bias

        sigma = torch.exp(m.log_noise).clamp(min=0.1)

        # Log-likelihood scaled to full dataset
        log_lik = dist.Normal(pred, sigma).log_prob(ratings).sum() * (n_total / B)

        # Subsampled KL, scaled to full dataset — unbiased estimator, much faster
        # than backpropping through all 279K users every step
        kl_u = 0.5 * (u_mu**2 + u_sig**2 - 2*torch.log(u_sig) - 1).sum() \
               * (m.num_users / len(unique_u))
        kl_b = 0.5 * (b_mu**2 + b_sig**2 - 2*torch.log(b_sig) - 1).sum() \
               * (m.num_businesses / len(unique_b))

        return -(log_lik - kl_u - kl_b)

    # ── training ─────────────────────────────────────────────────────────────

    def train(self, user_idx_np, business_idx_np, ratings_np,
              num_epochs=None, batch_size=4096, checkpoint_every=5,
              checkpoint_dir=None):
        import time
        from tqdm import tqdm
        from pathlib import Path

        if num_epochs is not None:
            self.num_epochs = num_epochs
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate)

        n = len(ratings_np)
        train_start = time.time()

        pbar = tqdm(range(self.num_epochs), desc="Bayesian VI", unit="epoch")
        for epoch in pbar:
            perm = np.random.permutation(n)
            u_shuf = user_idx_np[perm]
            b_shuf = business_idx_np[perm]
            r_shuf = ratings_np[perm]

            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, n, batch_size):
                u_b = torch.from_numpy(u_shuf[i:i+batch_size]).long()
                b_b = torch.from_numpy(b_shuf[i:i+batch_size]).long()
                r_b = torch.from_numpy(r_shuf[i:i+batch_size]).float()

                self._optimizer.zero_grad()
                loss = self._neg_elbo(u_b, b_b, r_b, n)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)
            elapsed = time.time() - train_start
            pbar.set_postfix(neg_elbo=f"{avg_loss:.0f}", elapsed=f"{elapsed:.0f}s")

            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                eta = elapsed / (epoch + 1) * (self.num_epochs - epoch - 1)
                tqdm.write(
                    f"\n--- Checkpoint epoch {epoch+1}/{self.num_epochs} ---"
                    f"\n  -ELBO:      {avg_loss:.2f}"
                    f"\n  Elapsed:    {elapsed:.0f}s  |  ETA: {eta:.0f}s\n"
                )
                if checkpoint_dir is not None:
                    ckpt = Path(checkpoint_dir) / f"bayesian_epoch{epoch+1:04d}.pt"
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'losses': self.losses,
                    }, ckpt)
                    tqdm.write(f"  Saved: {ckpt}\n")

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(self, user_idx, business_idx):
        """Predict using posterior means (MAP point estimate)."""
        with torch.no_grad():
            u_f = self.model.user_loc[user_idx]
            b_f = self.model.business_loc[business_idx]
            pred = (u_f * b_f).sum(-1) \
                 + self.model.user_bias[user_idx] \
                 + self.model.business_bias[business_idx] \
                 + self.model.global_bias
            return torch.clamp(pred, 1, 5).numpy()


class SimpleMatrixFactorization:
    """
    Simpler alternative: Non-probabilistic matrix factorization using SGD.
    Useful for quick experimentation before full Bayesian model.
    """
    
    def __init__(
        self,
        num_users: int,
        num_businesses: int,
        latent_dim: int = 20,
        learning_rate: float = 0.001,
        regularization: float = 0.01
    ):
        self.num_users = num_users
        self.num_businesses = num_businesses
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Initialize factors
        self.user_factors = np.random.randn(num_users, latent_dim) * 0.01
        self.business_factors = np.random.randn(num_businesses, latent_dim) * 0.01
        
        # Global bias terms
        self.user_bias = np.zeros(num_users)
        self.business_bias = np.zeros(num_businesses)
        self.global_bias = 0.0
        
        self.losses = []
    
    def train(self, user_idx, business_idx, ratings, num_epochs=100, batch_size=1024,
              checkpoint_every=10, checkpoint_dir=None):
        """
        Train using stochastic gradient descent.
        """
        import time
        from tqdm import tqdm
        from pathlib import Path

        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        num_samples = len(user_idx)
        self.global_bias = float(ratings.mean())
        train_start = time.time()

        pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        for epoch in pbar:
            epoch_start = time.time()
            perm = np.random.permutation(num_samples)
            u_shuf = user_idx[perm]
            b_shuf = business_idx[perm]
            r_shuf = ratings[perm]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, num_samples, batch_size):
                u_idx = u_shuf[i:i + batch_size]
                b_idx = b_shuf[i:i + batch_size]
                r = r_shuf[i:i + batch_size]

                # Snapshot factors before update so both gradients use the same values
                u_factors = self.user_factors[u_idx]   # [B, d]
                b_factors = self.business_factors[b_idx]  # [B, d]

                pred = (u_factors * b_factors).sum(axis=1)
                pred += self.user_bias[u_idx] + self.business_bias[b_idx] + self.global_bias
                error = pred - r  # [B]

                # Vectorized updates; np.add.at accumulates correctly for duplicate indices
                np.add.at(self.user_factors, u_idx,
                          -self.learning_rate * (error[:, None] * b_factors + self.regularization * u_factors))
                np.add.at(self.business_factors, b_idx,
                          -self.learning_rate * (error[:, None] * u_factors + self.regularization * b_factors))
                np.add.at(self.user_bias, u_idx, -self.learning_rate * error)
                np.add.at(self.business_bias, b_idx, -self.learning_rate * error)
                self.global_bias -= self.learning_rate * error.mean()

                loss = np.mean(error ** 2) + self.regularization * (
                    np.mean(self.user_factors ** 2) + np.mean(self.business_factors ** 2)
                )
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)
            elapsed = time.time() - train_start
            pbar.set_postfix(loss=f"{avg_loss:.4f}", elapsed=f"{elapsed:.0f}s")

            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                tqdm.write(
                    f"\n--- Checkpoint epoch {epoch + 1}/{num_epochs} ---"
                    f"\n  Loss:          {avg_loss:.4f}"
                    f"\n  MAE (approx):  {np.sqrt(avg_loss):.4f}"
                    f"\n  Elapsed:       {elapsed:.0f}s"
                    f"\n  ETA:           {elapsed / (epoch + 1) * (num_epochs - epoch - 1):.0f}s\n"
                )
                if checkpoint_dir is not None:
                    ckpt_path = Path(checkpoint_dir) / f"checkpoint_epoch{epoch + 1:04d}.npz"
                    np.savez(
                        ckpt_path,
                        user_factors=self.user_factors,
                        business_factors=self.business_factors,
                        user_bias=self.user_bias,
                        business_bias=self.business_bias,
                        global_bias=np.array([self.global_bias]),
                        losses=np.array(self.losses),
                    )
                    tqdm.write(f"  Saved checkpoint: {ckpt_path}\n")
    
    def predict(self, user_idx, business_idx):
        """Make predictions."""
        pred = (
            self.user_factors[user_idx] * self.business_factors[business_idx]
        ).sum(axis=1)
        pred += self.user_bias[user_idx] + self.business_bias[business_idx] + self.global_bias
        
        # Clip to valid range
        pred = np.clip(pred, 1, 5)
        
        return pred
