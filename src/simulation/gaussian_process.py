import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .firm import InvestmentParameters



class GPPrior(ABC):
    """Abstract base class for all GP Prior Mean beliefs."""
    def __init__(self, params: 'InvestmentParameters'):
        self.p = params

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluates the prior mean at points X.
        X must be an (N, 3) array where columns are (z, k, i).
        """
        pass

class ZeroPrior(GPPrior):
    """The naive assumption: the firm expects a value of 0 everywhere."""
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.zeros(X.shape[0])

class PerpetuityPrior(GPPrior):
    """
    The economic heuristic: the firm expects the steady-state dividend 
    for its current capital level forever, assuming maintenance investment.
    """
    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        z = X[:, 0]
        k = X[:, 1]
        
        k_safe = np.maximum(k, 1e-8)
        
        # Assume the firm just maintains its current capital
        i_maint = self.p.DELTA * k_safe
        
        adj_cost = 0.0
        if self.p.KAPPA > 0:
            adj_cost = (self.p.KAPPA / 2.0) * (i_maint**2 / k_safe)
            
        ss_dividend = z * (k_safe**self.p.ALPHA) - i_maint - adj_cost
        
        return ss_dividend / (1.0 - self.p.BETA)

class TrueValuePrior(GPPrior):
    """
    Wrapper for true value function. Must provide a callable that takes (z, k) and returns V*(z, k).
    """
    def __init__(self, params: 'InvestmentParameters', true_v_callable):
        super().__init__(params)
        self.get_true_v = true_v_callable

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        # Evaluate the true V*(z, k) for every row
        return np.array([self.get_true_v(z, k) for z, k, i in X])
    
@dataclass
class GPBeliefParameters:
    # RBF kernel hyperparameters
    sigma0: float = 1.0       # prior signal std
    lz: float = 0.5           # lengthscale in z
    lk: float = 3.0           # lengthscale in k
    li: float = 1.0           # lengthscale in i (investment)
    sigma_n: float = 0.01     # observation noise std (required for stability)


class GPBelief:
    def __init__(self, env_params: 'InvestmentParameters', gp_params: GPBeliefParameters, prior_mean_fn=None):
        self.env_params = env_params
        self.gp_params = gp_params
        
        # GP hyperparameters
        self.length_scales = np.array([gp_params.lz, gp_params.lk, gp_params.li])
        self.sigma0_sq = gp_params.sigma0 ** 2
        self.sigma_n_sq = gp_params.sigma_n ** 2 # noise variance to allow matrix inversion
        # prior mean
        self.prior_mean_fn = prior_mean_fn if prior_mean_fn else lambda x: np.zeros(x.shape[0])
        
        # history of observations (decision points, outcome points, and TD targets)
        self.X_dec = np.empty((0, 3))
        self.X_out = np.empty((0, 3))
        self.Y = np.empty(0)
        
        # Recursive Inverse Matrices
        self.C_inv = np.empty((0, 0))
        self.alpha = np.empty((0,))   # The weights vector for predictions

    def _kernel(self, X1, X2):
        """Calculates the RBF kernel matrix between two sets of points.
        Points must be (z, k, i) tuples.
        """
        X1_scaled = np.atleast_2d(X1) / self.length_scales
        X2_scaled = np.atleast_2d(X2) / self.length_scales
        
        # Efficient vectorized squared Euclidean distance
        sq_dist = (np.sum(X1_scaled**2, 1).reshape(-1, 1) + 
                   np.sum(X2_scaled**2, 1) - 
                   2 * np.dot(X1_scaled, X2_scaled.T))
        
        return self.sigma0_sq * np.exp(-0.5 * sq_dist)
    
    def _functional_cov(self, X_dec_1, X_out_1, X_dec_2, X_out_2):
        """
        Evaluates the 4-term covariance between two sets of TD transitions.
        Each transition is defined by a decision point and an outcome point.
        """
        beta = self.env_params.BETA
        
        k_dd = self._kernel(X_dec_1, X_dec_2)
        k_do = self._kernel(X_dec_1, X_out_2)
        k_od = self._kernel(X_out_1, X_dec_2)
        k_oo = self._kernel(X_out_1, X_out_2)
        
        return k_dd - beta * k_do - beta * k_od + (beta**2) * k_oo
    
    def _first_observation(self, x_dec, x_out, dividend):
        """Initializes the GP memory with the first empirical experience."""
        # Calculate novelty of the first point
        v_self = self._functional_cov(x_dec, x_out, x_dec, x_out)[0, 0]
        self.C_inv = np.array([[1.0 / (v_self + self.sigma_n_sq)]])
        
        # Store state
        self.X_dec = x_dec # Already forced to 2D in add_observation
        self.X_out = x_out
        self.Y = np.array([dividend])
        
        # Initial surprise (Innovation)
        m_0 = np.atleast_1d(self.prior_mean_fn(x_dec))[0] - self.env_params.BETA * np.atleast_1d(self.prior_mean_fn(x_out))[0]
        self.alpha = self.C_inv @ np.array([dividend - m_0])
    

    def add_observation(self, x_dec, x_out, dividend):
        """
        Incorporates a new Temporal Difference observation and recursively 
        updates the inverse Gram matrix and weights.
        """        
        # Ensure correct shape (1, 3)
        x_dec = np.atleast_2d(x_dec)
        x_out = np.atleast_2d(x_out)
        
        if len(self.Y) == 0:
            self._first_observation(x_dec, x_out, dividend)
            return
        
        v = self._functional_cov(x_dec, x_out, self.X_dec, self.X_out).T # covariance between new obs and history: shape (N, 1)
        w = self._functional_cov(x_dec, x_out, x_dec, x_out)[0, 0] + self.sigma_n_sq # variance of new obs + noise
        q = self.C_inv @ v
        S = w - (v.T @ q)[0, 0] # schur complement scalar
        
        # Build the new block matrix
        top_left = self.C_inv + (q @ q.T) / S
        top_right = -q / S
        bottom_left = -q.T / S
        bottom_right = np.array([[1.0 / S]])
        
        self.C_inv = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])

        # Update History
        self.X_dec = np.vstack([self.X_dec, x_dec])
        self.X_out = np.vstack([self.X_out, x_out])
        self.Y = np.append(self.Y, dividend)
        
        # Update Alpha (Batch approach - ensures numerical stability)
        M = self.prior_mean_fn(self.X_dec) - self.env_params.BETA * self.prior_mean_fn(self.X_out)
        self.alpha = self.C_inv @ (self.Y - M)

    def _predict_no_observations(self, X_query, return_std):
        """Returns the prior mean and variance when no observations have been made."""
        prior_q = self.prior_mean_fn(X_query)
        if return_std:
            return prior_q, np.sqrt(self.sigma0_sq) * np.ones(len(X_query))
        return prior_q

    def predict(self, X_query, return_std=False):
        """Predicts Q*(X_query) based on the accumulated linear functional observations."""
        X_q = np.atleast_2d(X_query)
        prior_q = self.prior_mean_fn(X_q) # Prior mean at query points
        
        # base case - if no observations, return prior mean and prior std
        if len(self.Y) == 0:
            return self._predict_no_observations(X_q, return_std)

        # Cross-covariance between query points and TD functionals
        K_dq = self._kernel(self.X_dec, X_q)
        K_oq = self._kernel(self.X_out, X_q)
        k_star = K_dq - self.env_params.BETA * K_oq # Shape: (N, N_queries)
        
        # Posterior Mean
        post_mean = prior_q + k_star.T @ self.alpha
        
        if not return_std:
            return post_mean
            
        # Posterior Variance
        K_qq_diag = np.full(len(X_q), self.sigma0_sq) # Diagonal of k(X_q, X_q)
        # Efficient quadratic form: diag(k_star^T * C_inv * k_star)
        variance_reduction = np.sum(k_star * (self.C_inv @ k_star), axis=0)
        post_var = K_qq_diag - variance_reduction
        
        return post_mean, np.sqrt(np.maximum(post_var, 0))