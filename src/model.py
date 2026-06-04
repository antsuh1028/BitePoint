"""
Probabilistic Graphical Model for Restaurant Recommendation.

This module implements a latent factor model using Pyro for Bayesian inference.
The model assumes:
- Each user has a latent preference vector
- Each restaurant has a latent feature vector
- Ratings are generated from the dot product of these vectors plus noise
"""

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
from config import MODEL_CONFIG, RANDOM_SEED

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class LatentFactorModel(nn.Module):
    """
    Latent Factor Model for rating prediction.
    
    The model uses:
    - User latent factors: z_u ~ N(0, I)
    - Business latent factors: z_b ~ N(0, I)
    - Rating: r ~ N(z_u · z_b, σ²)
    """
    
    def __init__(self, num_users: int, num_businesses: int, latent_dim: int = 20):
        super().__init__()
        self.num_users = num_users
        self.num_businesses = num_businesses
        self.latent_dim = latent_dim
        
        # Learnable parameters (variational parameters)
        self.user_loc = nn.Parameter(
            torch.randn(num_users, latent_dim) * 0.01
        )
        self.user_scale = nn.Parameter(
            torch.ones(num_users, latent_dim) * 0.1
        )
        
        self.business_loc = nn.Parameter(
            torch.randn(num_businesses, latent_dim) * 0.01
        )
        self.business_scale = nn.Parameter(
            torch.ones(num_businesses, latent_dim) * 0.1
        )
        
        self.noise_scale = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, user_idx, business_idx, ratings):
        """
        Forward pass: define the probabilistic model and likelihood.
        
        Args:
            user_idx: Tensor of user indices [batch_size]
            business_idx: Tensor of business indices [batch_size]
            ratings: Tensor of ratings [batch_size]
        """
        # Priors on latent factors
        with pyro.plate('user_plate', self.num_users):
            user_factors = pyro.sample(
                'user_factors',
                dist.Normal(self.user_loc, self.user_scale).to_event(1)
            )
        
        with pyro.plate('business_plate', self.num_businesses):
            business_factors = pyro.sample(
                'business_factors',
                dist.Normal(self.business_loc, self.business_scale).to_event(1)
            )
        
        # Observation model
        with pyro.plate('data', len(user_idx)):
            # Get factors for this batch
            u_factors = user_factors[user_idx]  # [batch_size, latent_dim]
            b_factors = business_factors[business_idx]  # [batch_size, latent_dim]
            
            # Compute predicted ratings (dot product)
            pred_ratings = (u_factors * b_factors).sum(-1)  # [batch_size]
            
            # Observation likelihood
            pyro.sample(
                'obs',
                dist.Normal(pred_ratings, torch.exp(self.noise_scale)),
                obs=ratings
            )


class VariationalInference:
    """Variational inference wrapper for the model."""
    
    def __init__(
        self,
        model: LatentFactorModel,
        learning_rate: float = 0.001,
        num_epochs: int = 100
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.losses = []
        
        # Set up optimization
        self.optimizer = Adam({'lr': learning_rate})
        self.svi = SVI(
            self.model,
            self._guide,
            self.optimizer,
            loss=Trace_ELBO()
        )
    
    def _guide(self, user_idx, business_idx, ratings):
        """
        Variational guide (approximation to posterior).
        This matches the structure of the forward pass.
        """
        # Variational distributions for latent factors
        with pyro.plate('user_plate', self.model.num_users):
            pyro.sample(
                'user_factors',
                dist.Normal(
                    self.model.user_loc,
                    torch.nn.functional.softplus(self.model.user_scale)
                ).to_event(1)
            )
        
        with pyro.plate('business_plate', self.model.num_businesses):
            pyro.sample(
                'business_factors',
                dist.Normal(
                    self.model.business_loc,
                    torch.nn.functional.softplus(self.model.business_scale)
                ).to_event(1)
            )
    
    def train(self, train_loader):
        """
        Train the model using stochastic variational inference.
        
        Args:
            train_loader: DataLoader with batches of (user_idx, business_idx, ratings)
        """
        print(f"Training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for user_idx, business_idx, ratings in train_loader:
                loss = self.svi.step(user_idx, business_idx, ratings)
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, user_idx, business_idx):
        """
        Make predictions on new data.
        
        Args:
            user_idx: Tensor of user indices
            business_idx: Tensor of business indices
            
        Returns:
            Predicted ratings
        """
        with torch.no_grad():
            # Get learned factors
            u_factors = self.model.user_loc[user_idx]  # [batch_size, latent_dim]
            b_factors = self.model.business_loc[business_idx]  # [batch_size, latent_dim]
            
            # Predict ratings
            pred_ratings = (u_factors * b_factors).sum(-1)
            
            # Clip to valid range [1, 5]
            pred_ratings = torch.clamp(pred_ratings, 1, 5)
        
        return pred_ratings.numpy()


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
    
    def train(self, user_idx, business_idx, ratings, num_epochs=100, batch_size=32):
        """
        Train using stochastic gradient descent.
        """
        print(f"Training for {num_epochs} epochs...")
        
        num_samples = len(user_idx)
        
        for epoch in range(num_epochs):
            # Shuffle data
            perm = np.random.permutation(num_samples)
            user_idx_shuffled = user_idx[perm]
            business_idx_shuffled = business_idx[perm]
            ratings_shuffled = ratings[perm]
            
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                # Get batch
                end_idx = min(i + batch_size, num_samples)
                u_idx = user_idx_shuffled[i:end_idx]
                b_idx = business_idx_shuffled[i:end_idx]
                r = ratings_shuffled[i:end_idx]
                
                # Forward pass
                pred = (
                    self.user_factors[u_idx] * self.business_factors[b_idx]
                ).sum(axis=1)
                pred += self.user_bias[u_idx] + self.business_bias[b_idx] + self.global_bias
                
                # Compute error
                error = pred - r
                
                # Update factors
                for j in range(end_idx - i):
                    u = u_idx[j]
                    b = b_idx[j]
                    e = error[j]
                    
                    # Gradient descent
                    self.user_factors[u] -= self.learning_rate * (
                        e * self.business_factors[b] +
                        self.regularization * self.user_factors[u]
                    )
                    self.business_factors[b] -= self.learning_rate * (
                        e * self.user_factors[u] +
                        self.regularization * self.business_factors[b]
                    )
                    self.user_bias[u] -= self.learning_rate * e
                    self.business_bias[b] -= self.learning_rate * e
                
                # Compute loss
                loss = np.mean(error ** 2) + self.regularization * (
                    np.mean(self.user_factors ** 2) +
                    np.mean(self.business_factors ** 2)
                )
                epoch_loss += loss
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def predict(self, user_idx, business_idx):
        """Make predictions."""
        pred = (
            self.user_factors[user_idx] * self.business_factors[business_idx]
        ).sum(axis=1)
        pred += self.user_bias[user_idx] + self.business_bias[business_idx] + self.global_bias
        
        # Clip to valid range
        pred = np.clip(pred, 1, 5)
        
        return pred
