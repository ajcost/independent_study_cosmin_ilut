from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.special import softmax

import quantecon as qe
from .gaussian_process import GPBelief # type: ignore[import]

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
    
class RationalInvestmentAgent(InvestmentAgent):

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
class InvestmentAgentParameters:
    H: float = 0.05 # exploration regularization parameter for ExperienceAgent

class ExperienceAgent(InvestmentAgent):
    def __init__(self, params: InvestmentParameters, gp: GPBelief, agent_params: InvestmentAgentParameters, seed: int = 42):
        super().__init__(params, name="Experience Agent")
        self.gp = gp
        self.agent_params = agent_params
        self.rng = np.random.default_rng(seed)
        
        # Grid used purely to define the discrete action space C for the softmax
        self.k_candidates = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)

    def _get_action_queries(self, z: float, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i_candidates = self.k_candidates - (1.0 - self.p.DELTA) * k # investments implied by candidate k' actions
    
        valid_mask = i_candidates >= 0.0 # boundary condition: can't disinvest beyond fully depreciating current capital
        
        valid_k = self.k_candidates[valid_mask]
        valid_i = i_candidates[valid_mask]
        
        # 3. Fallback: If the firm is massively overcapitalized (e.g., above the grid max)
        # the only physically possible action is to invest 0 and let it depreciate.
        if len(valid_k) == 0:
            valid_k = np.array([(1.0 - self.p.DELTA) * k])
            valid_i = np.array([0.0])
            
        # 4. Build the query matrix for the GP
        X_query = np.column_stack([
            np.full(len(valid_k), z),
            np.full(len(valid_k), k),
            valid_i
        ])
        
        return valid_k, valid_i, X_query

    def _entropy_policy(self, q: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Implements Proposition 2: The endogenous temperature softmax.
        Returns the probability distribution AND the shadow price delta_t.
        """
        N = len(q)
        H_max = np.log(N)
        
        # Uncertainty regularization
        lower_bound_entropy = self.agent_params.H * np.sum(std)

        # Base case 1: Uncertainty is basically zero -> Greedy Action - approximate delta as 0
        if lower_bound_entropy <= 1e-8:
            p = np.zeros(N)
            p[np.argmax(q)] = 1.0
            return p, 0.0

        # Base case 2: Uncertainty is massive -> Uniform Random - approximate delta as infinity
        if lower_bound_entropy >= H_max - 1e-8:
            return np.ones(N) / N, np.inf

        # Root finding function for delta
        def f(delta):
            # Numerically stable softmax
            logits = q / max(delta, 1e-12)
            logits -= np.max(logits)
            p = np.exp(logits)
            p /= np.sum(p)
            
            # Calculate entropy
            H = -np.sum(p * np.log(p + 1e-12))
            return H - lower_bound_entropy

        # Find the delta that satisfies the entropy constraint
        try:
            delta_star = brentq(f, 1e-6, 1e6, maxiter=200)
        except ValueError:
            warnings.warn("Brentq failed to find a root for delta. Defaulting to uniform random policy.")
            # If brentq fails to bracket, default to a high temperature
            delta_star = 1e3

        # Construct final probabilities using delta_star
        logits = q / max(delta_star, 1e-12)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= np.sum(p)
        
        return p, float(delta_star)

    def policy_kprime(self, z: float, k: float, t=None, return_delta=False) -> float:
        """Evaluates candidates via GP and samples from the entropy-constrained policy."""
        # 1. Get the DYNAMICALLY sized list of valid physical actions
        k_cands, i_cands, X_query = self._get_action_queries(z, k)
        
        # 2. GP Mental Simulation
        mean, std = self.gp.predict(X_query, return_std=True)
        
        # 3. Calculate endogenous policy and temperature
        probs, delta_t = self._entropy_policy(mean, std)
        
        # 4. FIX: Sample from the number of valid candidates, not the static N_k
        chosen_idx = self.rng.choice(len(k_cands), p=probs)
        chosen_kprime = float(k_cands[chosen_idx])
        
        if return_delta:
            return chosen_kprime, delta_t # pyright: ignore[reportReturnType]
        return chosen_kprime

    def get_greedy_action(self, z: float, k: float) -> float:
        """Used strictly for formulating the TD target."""
        k_cands, i_cands, X_query = self._get_action_queries(z, k)
        mean = self.gp.predict(X_query, return_std=False)
        return float(k_cands[np.argmax(mean)])

    def fixed_point(self) -> float:
        z = 1.0
        diffs = []
        for k in self.k_candidates:
            kp = self.get_greedy_action(z, k)
            diffs.append(abs(kp - k))
        return float(self.k_candidates[np.argmin(diffs)])