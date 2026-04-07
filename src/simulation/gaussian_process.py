import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .firm import InvestmentParameters, RationalInvestmentAgent, InvestmentEnvironment


class GPPrior(ABC):
    def __init__(self, env: 'InvestmentEnvironment'):
        self.p = env.p
        self.env = env

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        z, k, i = X[:, 0], X[:, 1], X[:, 2]
        k_next = (1.0 - self.p.DELTA) * k + i
        b_next = np.array([self.env.optimal_b_next(kn) for kn in k_next])

        flow = np.array([
            self.env.gp_observation(zz, kk, ii, bn)
            for zz, kk, ii, bn in zip(z, k, i, b_next)
        ])
        return flow + self.p.BETA * self._continuation(k_next)

    @abstractmethod
    def _continuation(self, k_next: np.ndarray) -> np.ndarray:
        pass


class ZeroPrior(GPPrior):
    def _continuation(self, k_next):
        return np.zeros_like(k_next)


class PerpetuityPrior(GPPrior):
    def _continuation(self, k_next):
        k_safe = np.maximum(k_next, 1e-8)
        i_maint = self.p.DELTA * k_safe
        adj = (self.p.KAPPA / 2.0) * (i_maint**2 / k_safe) if self.p.KAPPA > 0 else 0.0
        d_ss = k_safe**self.p.ALPHA - i_maint - adj  # z=1 at steady state
        return d_ss / (1.0 - self.p.BETA)

    
class TrueValueFunctionPrior(GPPrior):
    """
    A prior that decomposes Q into an exact flow payoff from the chosen
    investment action and a discounted continuation value at the resulting
    capital stock k' = (1 - delta)*k + i. The continuation value is
    approximated by a power-law fitted to the rational agent's value
    function V*(k) evaluated at z=1.
    """
    def __init__(self, rational_agent):
        super().__init__(rational_agent.env)
        iz_idx = int(np.argmin(np.abs(rational_agent.env.z_grid - 1.0)))
        self._v_interp = interp1d(
            rational_agent.env.k_grid,
            rational_agent.v[iz_idx, :],
            kind='cubic',
            fill_value='extrapolate'
        )

    def _continuation(self, k_next):
        return self._v_interp(np.maximum(k_next, 1e-8))

    def __call__(self, X):
        X = np.atleast_2d(X)
        z, k, i = X[:, 0], X[:, 1], X[:, 2]
        k_next = (1.0 - self.p.DELTA) * k + i
        b_next = np.array([self.env.optimal_b_next(kn) for kn in k_next])

        flow = np.array([
            self.env.gp_observation(zz, kk, ii, bn)
            for zz, kk, ii, bn in zip(z, k, i, b_next)
        ])
        return flow + self.p.BETA * self._continuation(k_next)

class Kernel(ABC):
    """Abstract base class for GP Kernels."""
    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Evaluate the kernel between two sets of points."""
        pass

    @abstractmethod
    def diag(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the diagonal of the kernel matrix for a set of points."""
        pass


class RBFKernel(Kernel):
    """Squared Exponential (RBF) Kernel for infinitely differentiable, smooth functions.
    
    NOTE: May be susceptible to Runge's phenomenon in extrapolation.
    """
    def __init__(self, sigma0: float, length_scales: list[float]):
        self.sigma0_sq = sigma0 ** 2
        self.length_scales = np.array(length_scales)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_scaled = np.atleast_2d(X1) / self.length_scales
        X2_scaled = np.atleast_2d(X2) / self.length_scales
        
        sq_dist = (np.sum(X1_scaled**2, 1).reshape(-1, 1) + 
                   np.sum(X2_scaled**2, 1) - 
                   2 * np.dot(X1_scaled, X2_scaled.T))
        
        # Clip to 0 to prevent tiny negative numbers from float precision issues
        sq_dist = np.maximum(sq_dist, 0.0) 
        
        return self.sigma0_sq * np.exp(-0.5 * sq_dist)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.sigma0_sq)


class LaplacianKernel(Kernel):
    """Laplacian (Matern 1/2) Kernel. Used in original Ilut & Vachev (2023) implementation."""
    def __init__(self, sigma0: float, length_scales: list[float]):
        self.sigma0_sq = sigma0 ** 2
        self.length_scales = np.array(length_scales)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_scaled = np.atleast_2d(X1) / self.length_scales
        X2_scaled = np.atleast_2d(X2) / self.length_scales
        
        # Calculate L2 distance without the square (Matern 1/2)
        sq_dist = (np.sum(X1_scaled**2, 1).reshape(-1, 1) + 
                   np.sum(X2_scaled**2, 1) - 
                   2 * np.dot(X1_scaled, X2_scaled.T))
        
        dist = np.sqrt(np.maximum(sq_dist, 0.0))
        return self.sigma0_sq * np.exp(-dist)

    def diag(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.sigma0_sq)

@dataclass
class GPBeliefParameters:
    kernel: Kernel
    sigma_n: float = 0.01  # observation noise std (required for stability)


class GPBelief:
    def __init__(self, env_params: 'InvestmentParameters', gp_params: GPBeliefParameters, prior_mean_fn: Callable = None):
        self.env_params = env_params
        self.gp_params = gp_params
        self.kernel = gp_params.kernel
        
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

    def _functional_cov(self, X_dec_1, X_out_1, X_dec_2, X_out_2):
        """
        Evaluates the 4-term covariance between two sets of TD transitions.
        Each transition is defined by a decision point and an outcome point.
        """
        beta = self.env_params.BETA
        
        # Delegated to the injected Kernel object
        k_dd = self.kernel(X_dec_1, X_dec_2)
        k_do = self.kernel(X_dec_1, X_out_2)
        k_od = self.kernel(X_out_1, X_dec_2)
        k_oo = self.kernel(X_out_1, X_out_2)
        
        return k_dd - beta * k_do - beta * k_od + (beta**2) * k_oo
    
    def _first_observation(self, x_dec, x_out, dividend):
        """Initializes the GP memory with the first empirical experience."""
        v_self = self._functional_cov(x_dec, x_out, x_dec, x_out)[0, 0]
        self.C_inv = np.array([[1.0 / (v_self + self.sigma_n_sq)]])
        
        self.X_dec = x_dec 
        self.X_out = x_out
        self.Y = np.array([dividend])
        
        m_0 = np.atleast_1d(self.prior_mean_fn(x_dec))[0] - self.env_params.BETA * np.atleast_1d(self.prior_mean_fn(x_out))[0]
        self.alpha = self.C_inv @ np.array([dividend - m_0])
    
    def add_observation(self, x_dec, x_out, dividend):
        """
        Incorporates a new Temporal Difference observation and recursively 
        updates the inverse Gram matrix and weights.
        """        
        x_dec = np.atleast_2d(x_dec)
        x_out = np.atleast_2d(x_out)
        
        if len(self.Y) == 0:
            self._first_observation(x_dec, x_out, dividend)
            return
        
        v = self._functional_cov(x_dec, x_out, self.X_dec, self.X_out).T 
        w = self._functional_cov(x_dec, x_out, x_dec, x_out)[0, 0] + self.sigma_n_sq 
        q = self.C_inv @ v
        S = max(w - (v.T @ q)[0, 0], 1e-10)
        
        top_left = self.C_inv + (q @ q.T) / S
        top_right = -q / S
        bottom_left = -q.T / S
        bottom_right = np.array([[1.0 / S]])
        
        self.C_inv = np.block([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])

        self.X_dec = np.vstack([self.X_dec, x_dec])
        self.X_out = np.vstack([self.X_out, x_out])
        self.Y = np.append(self.Y, dividend)
        
        M = self.prior_mean_fn(self.X_dec) - self.env_params.BETA * self.prior_mean_fn(self.X_out)
        self.alpha = self.C_inv @ (self.Y - M)

    def _predict_no_observations(self, X_query, return_std):
        """Returns the prior mean and variance when no observations have been made."""
        prior_q = self.prior_mean_fn(X_query)
        if return_std:
            # Use kernel's diag method to fetch base variance
            return prior_q, np.sqrt(self.kernel.diag(X_query))
        return prior_q
        
    def _k_star(self, X_q):
        K_dq = self.kernel(self.X_dec, X_q)
        K_oq = self.kernel(self.X_out, X_q)
        return K_dq - self.env_params.BETA * K_oq

    def predict(self, X_query, return_std=False):
        X_q = np.atleast_2d(X_query)
        if len(self.Y) == 0:
            return self._predict_no_observations(X_q, return_std)
        
        k_star = self._k_star(X_q)
        post_mean = self.prior_mean_fn(X_q) + k_star.T @ self.alpha
        
        if not return_std:
            return post_mean
        
        K_qq_diag = self.kernel.diag(X_q)
        variance_reduction = np.sum(k_star * (self.C_inv @ k_star), axis=0)
        post_var = K_qq_diag - variance_reduction
        return post_mean, np.sqrt(np.maximum(post_var, 0))

    def predict_full(self, X_query):
        X_q = np.atleast_2d(X_query)
        if len(self.Y) == 0:
            return self.prior_mean_fn(X_q), self.kernel(X_q, X_q)
        
        k_star = self._k_star(X_q)
        post_mean = self.prior_mean_fn(X_q) + k_star.T @ self.alpha
        
        K_qq = self.kernel(X_q, X_q)
        post_cov = K_qq - k_star.T @ self.C_inv @ k_star
        return post_mean, post_cov