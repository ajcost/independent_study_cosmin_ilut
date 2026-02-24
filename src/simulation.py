from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import softmax

import quantecon as qe # type: ignore[import]

@dataclass
class InvestmentParameters:
    ALPHA: float = 0.33 # Capital production elasticity
    DELTA: float = 0.04 # Depreciation rate
    R: float = 0.03 # Interest rate
    
    # These will be set based on the scenario
    KAPPA: float = 0.1 # Adjustment cost parameter
    RHO: float = 0.9 # Persistence
    SIGMA_EPS: float = 0.0 # Volatility (0 = Deterministic)

    # Grid Parameters
    N_k: int = 100 # Number of capital grid points
    N_z: int = 5 # Number of shock states (if stochastic)
    K_min: float = 0.1 # Minimum capital
    K_max: float = 20.0 # Maximum capital

    # Calculated post-init
    BETA: float = field(init=False) # Discount factor

    def __post_init__(self):
        self.BETA = 1 / (1 + self.R)

@dataclass
class SimResult:
    t: np.ndarray
    z: np.ndarray
    k: np.ndarray
    k_next: np.ndarray
    i: np.ndarray
    d: np.ndarray


class InvestmentEnvironment:
    def __init__(self, params: InvestmentParameters, seed=0):
        self.p = params
        self.rng = np.random.default_rng(seed)

    def production(self, z, k):
        return z * (k ** self.p.ALPHA)

    def dividend(self, z, k, i):
        k_safe = max(k, 1e-8)
        adj_cost = 0.0
        if self.p.KAPPA > 0:
            adj_cost = (self.p.KAPPA / 2.0) * (i**2 / k_safe)
        return self.production(z, k_safe) - i - adj_cost

    def get_reward(self, z, k, k_prime):
        # action is k' = next capital
        k_safe = max(k, 1e-8)
        k_prime = max(k_prime, 1e-8)

        i = k_prime - (1.0 - self.p.DELTA) * k_safe
        return self.dividend(z, k_safe, i)

    def transition(self, z, k_prime, custom_rng=None):
        k_next = max(k_prime, 1e-8)

        if self.p.SIGMA_EPS > 0:
            eps = self.rng.normal(0.0, self.p.SIGMA_EPS) if custom_rng is None else custom_rng.normal(0.0, self.p.SIGMA_EPS)
            z_next = np.exp(self.p.RHO * np.log(max(z, 1e-12)) + eps)
        else:
            z_next = z

        return z_next, k_next

class InvestmentAgent(ABC):
    """Abstract base class for an investment agent. Subclasses should implement policy_kprime.
    """

    def __init__(self, params: InvestmentParameters, name: str):
        self.p = params
        self.name = name

    def fit(self):
        """Optional training/solving step. Default: do nothing."""
        return self

    @abstractmethod
    def policy_kprime(self, z: float, k: float, t: int | None = None) -> float:
        """Given state (z,k), return chosen next capital k'.

        Arguments:
            z(float): Current shock state.
            k(float): Current capital.
            t(int|None): Current time step (optional, for non-stationary policies).

        Returns:
            k'(float): Chosen next capital.
        """
        raise NotImplementedError
    
    @abstractmethod
    def fixed_point(self) -> float:
        """Return the fixed point k' = k for the agent's policy. Used for steady state analysis."""
        raise NotImplementedError

    def simulate(self, env: InvestmentEnvironment, T=60, z0=1.0, k0=10.0, seed=0) -> SimResult:
        """Unified simulation harness. Uses env.get_reward + env.transition.

        Arguments:
            env(InvestmentEnvironment): The environment to simulate in.
            T(int): Number of time steps to simulate.
            z0(float): Initial shock state.
            k0(float): Initial capital.
            seed(int): Random seed for reproducibility.

        Returns:
            SimResult: Dataclass containing simulation results.
        """
        rng = np.random.default_rng(seed)

        t = np.arange(T, dtype=int)
        z = np.empty(T, dtype=float)
        k = np.empty(T, dtype=float)
        k_next = np.empty(T, dtype=float)
        i = np.empty(T, dtype=float)
        d = np.empty(T, dtype=float)

        z[0] = float(z0)
        k[0] = float(k0)

        for tt in range(T):
            # choose action
            kp = float(self.policy_kprime(z[tt], k[tt], t=tt))
            k_next[tt] = kp

            # implied investment
            i[tt] = kp - (1.0 - self.p.DELTA) * k[tt]

            # dividend (reward)
            d[tt] = env.get_reward(z[tt], k[tt], kp)

            # transition (except last)
            if tt < T - 1:
                z_next, k_state_next = env.transition(z[tt], kp, custom_rng=rng)
                z[tt + 1] = z_next
                k[tt + 1] = k_state_next

        return SimResult(t=t, z=z, k=k, k_next=k_next, i=i, d=d)
    

    def simulate_irf(
        self,
        env: InvestmentEnvironment,
        T: int = 40,
        shock_size_log: float = 0.05,
        shock_time: int = 0,
        z0: float = 1.0,
        k0: float | None = None,
        deterministic_after: bool = True,
        seed: int = 0,
    ) -> dict:
        """
        Impulse response for any agent that implements policy_kprime() and fixed_point().

        Convention:
          - Apply a one-time impulse to log(z) at shock_time.
          - If deterministic_after=True: set future innovations eps_{t>shock_time}=0 (standard DSGE IRF).
          - If deterministic_after=False: simulate future shocks with common random numbers (Monte Carlo-style),
            using the env.transition() RNG (seeded).

        Returns:
          dict with baseline and shocked SimResult plus diff and percent deviations.
        """
        p = self.p

        if k0 is None:
            k0 = float(self.fixed_point())

        tgrid = np.arange(T, dtype=int)

        rng_base = np.random.default_rng(seed)
        rng_shk = np.random.default_rng(seed)  # common random numbers

        def step_z(z_now: float, rng) -> float:
            if p.SIGMA_EPS <= 0:
                return z_now
            if deterministic_after:
                # eps = 0
                return float(np.exp(p.RHO * np.log(max(z_now, 1e-12))))
            # stochastic eps
            eps = float(rng.normal(0.0, p.SIGMA_EPS))
            return float(np.exp(p.RHO * np.log(max(z_now, 1e-12)) + eps))

        def run_path(apply_shock: bool, rng) -> SimResult:
            z = np.empty(T, dtype=float)
            k = np.empty(T, dtype=float)
            k_next = np.empty(T, dtype=float)
            i = np.empty(T, dtype=float)
            d = np.empty(T, dtype=float)

            z[0] = float(z0)
            k[0] = float(k0)

            # apply impulse at t=0
            if apply_shock and shock_time == 0:
                z[0] = float(np.exp(np.log(max(z[0], 1e-12)) + shock_size_log))

            for tt in range(T):
                kp = float(self.policy_kprime(z[tt], k[tt], t=tt))
                k_next[tt] = kp
                i[tt] = kp - (1.0 - p.DELTA) * k[tt]
                d[tt] = float(env.get_reward(z[tt], k[tt], kp))

                if tt < T - 1:
                    # capital transition is deterministic given action
                    k[tt + 1] = max(kp, 1e-8)

                    # productivity transition
                    z[tt + 1] = step_z(z[tt], rng)

                    # apply impulse at shock_time if not 0
                    if apply_shock and (tt + 1) == shock_time:
                        z[tt + 1] = float(np.exp(np.log(max(z[tt + 1], 1e-12)) + shock_size_log))

            return SimResult(t=tgrid, z=z, k=k, k_next=k_next, i=i, d=d)

        base = run_path(False, rng_base)
        shk  = run_path(True, rng_shk)

        diff = SimResult(
            t=tgrid,
            z=shk.z - base.z,
            k=shk.k - base.k,
            k_next=shk.k_next - base.k_next,
            i=shk.i - base.i,
            d=shk.d - base.d,
        )

        pct = {
            "z": 100.0 * (shk.z / base.z - 1.0),
            "k": 100.0 * (shk.k / base.k - 1.0),
            "i": 100.0 * (shk.i / base.i - 1.0),
            "d": 100.0 * (shk.d / base.d - 1.0),
        }

        return {"t": tgrid, "base": base, "shk": shk, "diff": diff, "pct": pct}

    def policy_curve(self, z_grid: np.ndarray, k_grid: np.ndarray, z_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return arrays (k_grid, kprime_grid) for plotting k -> k' at a given z index.
        
        By default uses the agent's policy_kprime on grid points.

        Arguments:
            z_grid(np.ndarray): Grid of z values.
            k_grid(np.ndarray): Grid of k values.
            z_idx(int): Index of z_grid to use for plotting.

        Returns:
            (k_grid, kprime_grid): Tuple of arrays for plotting.
        """
        z_val = float(z_grid[z_idx])
        kprime_vals = np.array([self.policy_kprime(z_val, float(kv)) for kv in k_grid], dtype=float)
        return k_grid, kprime_vals
    
class RationalInvestment(InvestmentAgent):

    def __init__(self, params: InvestmentParameters):
        super().__init__(params, name="Rational (VFI)")
        self.setup_grids()
        self.initialize_value_function()

    def setup_grids(self):
        if self.p.SIGMA_EPS > 0 and self.p.N_z > 1:
            mc = qe.markov.approximation.tauchen(
                self.p.N_z, self.p.RHO, self.p.SIGMA_EPS, mu=0, n_std=3
            )
            self.z_grid = np.exp(mc.state_values)
            self.P = mc.P
            self.actual_nz = self.p.N_z
        else:
            self.z_grid = np.array([1.0])
            self.P = np.array([[1.0]])
            self.actual_nz = 1

        self.k_grid = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)

    def initialize_value_function(self):
        self.v = np.zeros((self.actual_nz, self.p.N_k))
        self.policy_k = np.zeros((self.actual_nz, self.p.N_k))

        for i_z in range(self.actual_nz):
            output = self.z_grid[i_z] * (self.k_grid ** self.p.ALPHA)
            self.v[i_z, :] = output / self.p.R

    def get_dividend(self, k_next, k_current, z):
        investment = k_next - (1 - self.p.DELTA) * k_current
        adj_cost = 0.0
        if self.p.KAPPA > 0:
            adj_cost = (self.p.KAPPA / 2) * (investment**2 / k_current)
        return z * (k_current ** self.p.ALPHA) - investment - adj_cost

    def fit(self, tol=1e-5, max_iter=500):
        for _ in range(max_iter):
            v_new = np.zeros_like(self.v)
            policy_new = np.zeros_like(self.policy_k)

            v_expected = self.P @ self.v

            ev_funcs = [
                interp1d(self.k_grid, v_expected[i, :],
                         kind="linear", fill_value="extrapolate")  # type: ignore[reportUnknownMemberType]
                for i in range(self.actual_nz)
            ]

            for i_z in range(self.actual_nz):
                z = self.z_grid[i_z]
                cont_func = ev_funcs[i_z]

                for i_k, k in enumerate(self.k_grid):

                    def objective(k_prime):
                        div = self.get_dividend(k_prime, k, z)
                        return -(div + self.p.BETA * cont_func(k_prime))

                    res = minimize_scalar(
                        objective,
                        bounds=(self.p.K_min, self.p.K_max),
                        method="bounded"
                    )

                    v_new[i_z, i_k] = -res.fun
                    policy_new[i_z, i_k] = res.x

            if np.max(np.abs(v_new - self.v)) < tol:
                break

            self.v = v_new
            self.policy_k = policy_new

        # build fast interpolators for policy queries
        self._pol_funcs = [
            interp1d(self.k_grid, self.policy_k[i, :],
                     kind="linear", fill_value="extrapolate")  # type: ignore[reportUnknownMemberType]
            for i in range(self.actual_nz)
        ]

        return self

    def _z_to_idx(self, z):
        return int(np.argmin(np.abs(self.z_grid - z)))

    def policy_kprime(self, z: float, k: float, t=None) -> float:
        z_idx = self._z_to_idx(z)
        return float(self._pol_funcs[z_idx](k))
    
    def fixed_point(self) -> float:
        def objective(k):
            return self.policy_kprime(1.0, k) - k  # z=1 for deterministic

        k_min = self.k_grid[0]
        k_max = self.k_grid[-1]

        return brentq(objective, k_min, k_max) # pyright: ignore[reportReturnType]

@dataclass
class GPBeliefParameters:
    # RBF kernel hyperparameters
    sigma0: float = 1.0       # prior signal std
    lz: float = 0.5           # lengthscale in z
    lk: float = 3.0           # lengthscale in k
    li: float = 1.0           # lengthscale in i (investment)
    sigma_n: float = 0.01     # observation noise std (required for stability)


class GPBelief:
    def __init__(self, env_params: InvestmentParameters, gp_params: GPBeliefParameters, prior_mean_fn=None):
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


# class ExperienceReasoningAgent(InvestmentAgent):

#     def __init__(self, params: InvestmentParameters,
#                  gp_config: GPBeliefConfig,
#                  h_min: float = 0.05,
#                  reasoning_draws: int = 5,
#                  seed: int = 0):

#         super().__init__(params, name="Experience+Reasoning")
#         self.rng = np.random.default_rng(seed)
#         self.h_min = h_min
#         self.reasoning_draws = reasoning_draws

#         # build grids exactly like rational agent
#         if self.p.SIGMA_EPS > 0 and self.p.N_z > 1:
#             mc = qe.markov.approximation.tauchen(
#                 self.p.N_z, self.p.RHO, self.p.SIGMA_EPS, mu=0, n_std=3
#             )
#             self.z_grid = np.exp(mc.state_values)
#         else:
#             self.z_grid = np.array([1.0])

#         self.k_grid = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)

#         self.gp = GridGPBelief(self.z_grid, self.k_grid, gp_config)

#     def _entropy_policy(self, q: np.ndarray, var: np.ndarray) -> np.ndarray:
#         """
#         Robust entropy-constrained softmax.

#         Enforces:
#           - var cleaned to finite, >= jitter
#           - target entropy capped at log(N)
#           - if infeasible => uniform policy
#           - output probs finite and sums to 1
#         """
#         q = np.asarray(q, dtype=float)
#         var = np.asarray(var, dtype=float)

#         # 1) clean variances
#         var = np.where(np.isfinite(var), var, self.gp.cfg.jitter)
#         var = np.maximum(var, self.gp.cfg.jitter)

#         N = q.size
#         H_max = float(np.log(N))

#         target = float(self.h_min * np.sum(np.sqrt(var)))

#         # If target is infeasible, return uniform
#         if target >= H_max - 1e-10:
#             return np.ones(N, dtype=float) / N

#         # If target ~0, return greedy
#         if target <= 1e-10:
#             p = np.zeros(N, dtype=float)
#             p[int(np.argmax(q))] = 1.0
#             return p

#         def entropy_of_delta(delta: float) -> float:
#             delta = max(delta, 1e-12)
#             logits = q / delta
#             logits = logits - np.max(logits)  # stabilize
#             p = np.exp(logits)
#             s = float(np.sum(p))
#             if not np.isfinite(s) or s <= 0.0:
#                 return -np.inf
#             p = p / s
#             H = -float(np.sum(p * np.log(p + 1e-12)))
#             return H

#         def f(delta: float) -> float:
#             H = entropy_of_delta(delta)
#             return H - target

#         # Bracket: small delta => near-greedy low entropy, large delta => near-uniform high entropy
#         # Use a wide upper bound; if still can't hit, fall back to uniform.
#         lo, hi = 1e-6, 1e6

#         flo = f(lo)
#         fhi = f(hi)

#         if not (np.isfinite(flo) and np.isfinite(fhi)):
#             return np.ones(N, dtype=float) / N

#         # If no sign change, pick the best feasible endpoint
#         if flo > 0:
#             # even very small delta has too much entropy (rare); go greedy-ish
#             delta_star = lo
#         elif fhi < 0:
#             # even huge delta can't reach target (shouldn't happen if target < logN),
#             # but just return uniform safely
#             return np.ones(N, dtype=float) / N
#         else:
#             delta_star = float(brentq(f, lo, hi, maxiter=200))

#         # build probs
#         logits = q / max(delta_star, 1e-12)
#         logits = logits - np.max(logits)
#         p = np.exp(logits)
#         s = float(np.sum(p))

#         if not np.isfinite(s) or s <= 0.0:
#             return np.ones(N, dtype=float) / N

#         p = p / s

#         # final sanitization
#         p = np.where(np.isfinite(p), p, 0.0)
#         s2 = float(np.sum(p))
#         if s2 <= 0.0:
#             return np.ones(N, dtype=float) / N
#         return p / s2

#     def policy_kprime(self, z: float, k: float, t=None) -> float:
#         z_idx = self.gp.z_to_idx(z)
#         k_idx = self.gp.k_to_idx(k)

#         q = self.gp.q_vector(z_idx, k_idx)
#         var = self.gp.var_vector(z_idx, k_idx)

#         probs = self._entropy_policy(q, var)
#         kp_idx = self.rng.choice(self.p.N_k, p=probs)
#         return float(self.k_grid[kp_idx])

#     def fixed_point(self) -> float:
#         z = 1.0
#         diffs = []
#         for k in self.k_grid:
#             kp = self.policy_kprime(z, k)
#             diffs.append(kp - k)
#         diffs = np.array(diffs)
#         idx = np.argmin(np.abs(diffs))
#         return float(self.k_grid[idx])

#     def train(self, env: InvestmentEnvironment, T=5000, z0=1.0, k0=None):
#         if k0 is None:
#             k0 = float(self.k_grid[len(self.k_grid)//2])

#         z = z0
#         k = k0

#         for _ in range(T):
#             kp = self.policy_kprime(z, k)

#             r = env.get_reward(z, k, kp)
#             z_next, k_next = env.transition(z, kp)

#             # TD error
#             z_idx_next = self.gp.z_to_idx(z_next)
#             k_idx_next = self.gp.k_to_idx(k_next)
#             q_next_best = self.gp.q_best(z_idx_next, k_idx_next)

#             z_idx = self.gp.z_to_idx(z)
#             k_idx = self.gp.k_to_idx(k)
#             kp_idx = self.gp.k_to_idx(kp)

#             td = r + self.p.BETA * q_next_best - self.gp.mu[z_idx, k_idx, kp_idx]

#             # experience update
#             self.gp.update_experience_td(z, k, kp, td)

#             # reasoning updates
#             for _ in range(self.reasoning_draws):
#                 kp_sim = float(self.rng.choice(self.k_grid))
#                 r_sim = env.get_reward(z, k, kp_sim)

#                 z_next_mean = np.exp(self.p.RHO * np.log(max(z, 1e-12)))
#                 z_idx_m = self.gp.z_to_idx(z_next_mean)
#                 k_idx_m = self.gp.k_to_idx(kp_sim)

#                 q_future = self.gp.q_best(z_idx_m, k_idx_m)
#                 y = r_sim + self.p.BETA * q_future + self.rng.normal(0, self.gp.cfg.sigma_reason)

#                 self.gp.update_reasoning(z, k, kp_sim, y)

#             z, k = z_next, k_next

#         return self