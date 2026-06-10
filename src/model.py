import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from config import MODEL_CONFIG, RANDOM_SEED

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class LatentFactorModel(nn.Module):
    def __init__(self, num_users, num_businesses, latent_dim=20, global_bias=3.75):
        super().__init__()
        self.num_users = num_users
        self.num_businesses = num_businesses
        self.latent_dim = latent_dim

        self.user_loc   = nn.Parameter(torch.randn(num_users, latent_dim) * 0.01)
        self.user_scale = nn.Parameter(torch.ones(num_users, latent_dim) * 0.5)

        self.business_loc   = nn.Parameter(torch.randn(num_businesses, latent_dim) * 0.01)
        self.business_scale = nn.Parameter(torch.ones(num_businesses, latent_dim) * 0.5)

        self.user_bias     = nn.Parameter(torch.zeros(num_users))
        self.business_bias = nn.Parameter(torch.zeros(num_businesses))
        self.global_bias   = nn.Parameter(torch.tensor(global_bias))
        self.log_noise     = nn.Parameter(torch.tensor(0.0))


class VariationalInference:
    def __init__(self, model, learning_rate=0.01, num_epochs=20):
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.losses = []
        self._optimizer = None

    #lecture 08b - Variational
    #ELBO = E_q[log p(x|z)] - KL(q(z) || p(z))
    def _neg_elbo(self, u_idx, b_idx, ratings):
        m = self.model

        u_mu  = m.user_loc[u_idx]
        u_sig = nn.functional.softplus(m.user_scale[u_idx]) + 1e-6
        b_mu  = m.business_loc[b_idx]
        b_sig = nn.functional.softplus(m.business_scale[b_idx]) + 1e-6

        u_z = u_mu + u_sig * torch.randn_like(u_mu)
        b_z = b_mu + b_sig * torch.randn_like(b_mu)

        pred = (u_z * b_z).sum(-1) + m.user_bias[u_idx] + m.business_bias[b_idx] + m.global_bias
        sigma = torch.exp(m.log_noise).clamp(min=0.1)

        log_lik = dist.Normal(pred, sigma).log_prob(ratings).sum()
        kl_u = 0.5 * (u_mu**2 + u_sig**2 - 2*torch.log(u_sig) - 1).sum()
        kl_b = 0.5 * (b_mu**2 + b_sig**2 - 2*torch.log(b_sig) - 1).sum()

        return -(log_lik - kl_u - kl_b)

    def train(self, user_idx_np, business_idx_np, ratings_np, num_epochs=None, batch_size=4096):
        import time
        from tqdm import tqdm

        if num_epochs is not None:
            self.num_epochs = num_epochs
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        n = len(ratings_np)
        train_start = time.time()

        pbar = tqdm(range(self.num_epochs), desc="Bayesian VI", unit="epoch")
        for _ in pbar:
            perm  = np.random.permutation(n)
            u_shuf = user_idx_np[perm]
            b_shuf = business_idx_np[perm]
            r_shuf = ratings_np[perm]

            epoch_loss, num_batches = 0.0, 0
            for i in range(0, n, batch_size):
                u_b = torch.from_numpy(u_shuf[i:i+batch_size]).long()
                b_b = torch.from_numpy(b_shuf[i:i+batch_size]).long()
                r_b = torch.from_numpy(r_shuf[i:i+batch_size]).float()

                self._optimizer.zero_grad()
                loss = self._neg_elbo(u_b, b_b, r_b)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)
            elapsed = time.time() - train_start
            pbar.set_postfix(neg_elbo=f"{avg_loss:.0f}", elapsed=f"{elapsed:.0f}s")

    def predict(self, user_idx, business_idx):
        with torch.no_grad():
            pred = (self.model.user_loc[user_idx] * self.model.business_loc[business_idx]).sum(-1) \
                 + self.model.user_bias[user_idx] \
                 + self.model.business_bias[business_idx] \
                 + self.model.global_bias
            return torch.clamp(pred, 1, 5).numpy()


class SimpleMatrixFactorization:
    def __init__(self, num_users, num_businesses, latent_dim=20,
                 learning_rate=0.001, regularization=0.01):
        self.num_users = num_users
        self.num_businesses = num_businesses
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization

        self.user_factors     = np.random.randn(num_users, latent_dim) * 0.01
        self.business_factors = np.random.randn(num_businesses, latent_dim) * 0.01
        self.user_bias        = np.zeros(num_users)
        self.business_bias    = np.zeros(num_businesses)
        self.global_bias      = 0.0
        self.losses           = []

    def train(self, user_idx, business_idx, ratings, num_epochs=100, batch_size=1024):
        import time
        from tqdm import tqdm

        n = len(user_idx)
        self.global_bias = float(ratings.mean())
        train_start = time.time()

        pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
        for _ in pbar:
            perm  = np.random.permutation(n)
            u_shuf = user_idx[perm]
            b_shuf = business_idx[perm]
            r_shuf = ratings[perm]

            epoch_loss, num_batches = 0.0, 0
            for i in range(0, n, batch_size):
                u_b = u_shuf[i:i+batch_size]
                b_b = b_shuf[i:i+batch_size]
                r   = r_shuf[i:i+batch_size]

                u_f   = self.user_factors[u_b]
                b_f   = self.business_factors[b_b]
                pred  = (u_f * b_f).sum(axis=1) + self.user_bias[u_b] + self.business_bias[b_b] + self.global_bias
                error = pred - r

                np.add.at(self.user_factors,     u_b, -self.learning_rate * (error[:, None] * b_f + self.regularization * u_f))
                np.add.at(self.business_factors, b_b, -self.learning_rate * (error[:, None] * u_f + self.regularization * b_f))
                np.add.at(self.user_bias,        u_b, -self.learning_rate * error)
                np.add.at(self.business_bias,    b_b, -self.learning_rate * error)
                self.global_bias -= self.learning_rate * error.mean()

                loss = np.mean(error**2) + self.regularization * (
                    np.mean(self.user_factors**2) + np.mean(self.business_factors**2)
                )
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)
            elapsed = time.time() - train_start
            pbar.set_postfix(loss=f"{avg_loss:.4f}", elapsed=f"{elapsed:.0f}s")

    def predict(self, user_idx, business_idx):
        pred = (self.user_factors[user_idx] * self.business_factors[business_idx]).sum(axis=1)
        pred += self.user_bias[user_idx] + self.business_bias[business_idx] + self.global_bias
        return np.clip(pred, 1, 5)
