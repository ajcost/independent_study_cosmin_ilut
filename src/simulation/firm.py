from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar

import quantecon as qe
from .gaussian_process import GPBelief # type: ignore[import]

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gaussian_process import GPBeliefParameters

@dataclass
class InvestmentParameters:
    ALPHA: float = 0.33 # Capital production elasticity
    DELTA: float = 0.04 # Depreciation rate
    R: float = 0.03     # Interest rate
    THETA: float = 0.0  # Collateral constraint fraction (b_{t+1} <= theta * k_{t+1})
    BETA: float = 0.96  # Discount factor (set independently of R)

    # These will be set based on the scenario
    KAPPA: float = 0.1   # Adjustment cost parameter
    RHO: float = 0.9     # Persistence
    SIGMA_EPS: float = 0.0 # Volatility (0 = Deterministic)

    # Grid Parameters
    N_k: int = 100
    N_z: int = 7
    K_min: float = 0.01
    K_max: float = 20.0


@dataclass
class SimResult:
    t: np.ndarray
    z: np.ndarray
    k: np.ndarray
    k_next: np.ndarray
    i: np.ndarray
    d: np.ndarray
    b: np.ndarray
    b_next: np.ndarray


class AdjustmentCosts(ABC):
    @abstractmethod
    def __call__(self, i: float, k: float) -> float:
        """Returns psi(i, k)."""
        raise NotImplementedError

class NoAdjustmentCosts(AdjustmentCosts):
    def __call__(self, i: float, k: float) -> float:
        return 0.0

class QuadraticAdjustmentCosts(AdjustmentCosts):
    def __init__(self, kappa: float):
        self.kappa = kappa

    def __call__(self, i: float, k: float) -> float:
        k_safe = max(k, 1e-8)
        return (self.kappa / 2.0) * (i**2 / k_safe)


class InvestmentEnvironment:
    def __init__(self, params: InvestmentParameters, adjustment_costs: AdjustmentCosts, seed: int = 42):
        self.p = params
        self.adjustment_costs = adjustment_costs
        self.rng = np.random.default_rng(seed)
        self.setup_grids()

    def setup_grids(self):
        if self.p.SIGMA_EPS > 0 and self.p.N_z > 1:
            mc = qe.markov.approximation.tauchen(
                self.p.N_z, self.p.RHO, self.p.SIGMA_EPS, mu=0, n_std=3
            )
            self.z_grid = np.exp(mc.state_values)
            self.P = mc.P
            self.actual_nz = self.p.N_z
        else:
            warnings.warn("Zero volatility or zero z states: using degenerate z grid with a single point at 1.0.")
            self.z_grid = np.array([1.0])
            self.P = np.array([[1.0]])
            self.actual_nz = 1

        self.k_grid = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)

    def action_query_grid(self, z_t, k_t):
        """Returns a grid of (z, k, i) points for querying the GP given current state (z_t, k_t).
        Generally used to discretize the action space for the ExperienceReasoningAgent's policy.
        """
        i_grid = self.k_grid - (1 - self.p.DELTA) * k_t
        return np.column_stack([
            np.full_like(i_grid, z_t),
            np.full_like(i_grid, k_t),
            i_grid
        ])

    def production(self, z, k):
        return z * (k ** self.p.ALPHA)

    def optimal_b_next(self, k_next: float) -> float:
        """Optimal next-period debt given k_next.

        With linear utility, the net PV of a unit of debt is (1 - beta*(1+R)):
          - beta*(1+R) <= 1: borrow at collateral constraint
          - beta*(1+R) >  1: don't borrow
        """
        if self.p.BETA * (1.0 + self.p.R) <= 1.0:
            return self.p.THETA * k_next
        return 0.0

    def dividend(self, z, k, i, b=0.0, b_next=0.0):
        """d = f(z,k) - i - psi(i,k) + b_{t+1} - (1+R)*b_t"""
        k_safe = max(k, 1e-8)
        adj_cost = self.adjustment_costs(i, k_safe)
        return self.production(z, k_safe) - i - adj_cost + b_next - (1 + self.p.R) * b

    # TODO: Double check this
    def gp_observation(self, z, k, i, b_next):
        """Dividend corrected for debt terms for GP updates.

        The GP learns Q(z,k,b,i) = GP(z,k,i) - (1+R)*b.
        The correct GP observation target is:
            d + (1+R)*b - beta*(1+R)*b_next = f(z,k) - i - psi(i,k) + b_next*(1 - beta*(1+R))
        """
        k_safe = max(k, 1e-8)
        adj_cost = self.adjustment_costs(i, k_safe)
        correction = b_next * (1.0 - self.p.BETA * (1.0 + self.p.R))
        return self.production(z, k_safe) - i - adj_cost + correction

    def transition(self, z, k_prime, b_next=0.0, custom_rng=None):
        k_next = max(k_prime, 1e-8)
        if self.p.SIGMA_EPS > 0:
            rng = custom_rng if custom_rng is not None else self.rng
            eps = rng.normal(0.0, self.p.SIGMA_EPS)
            z_next = np.exp(self.p.RHO * np.log(max(z, 1e-12)) + eps)
        else:
            z_next = z
        return z_next, k_next, float(b_next)


class InvestmentAgent(ABC):
    """Abstract base class for an investment agent."""

    def __init__(self, env: InvestmentEnvironment, name: str):
        self.env = env
        self.p = env.p   # convenience shorthand
        self.name = name

    def fit(self):
        """Optional training/solving step. Default: do nothing."""
        return self

    @abstractmethod
    def policy(self, z: float, k: float, b: float = 0.0, t: int | None = None) -> tuple[float, float]:
        """Given state (z, k, b), return (k_next, b_next)."""
        raise NotImplementedError

    @abstractmethod
    def fixed_point(self) -> float:
        """Return the steady-state capital k* (with z=1). b* = THETA * k*."""
        raise NotImplementedError

    def simulate(self, env: InvestmentEnvironment | None = None, T=60, z0=1.0, k0=10.0, b0=0.0, seed=0) -> SimResult:
        """Simulate the agent. Uses the agent's own environment if env is not provided."""
        env = env or self.env
        rng = np.random.default_rng(seed)

        t = np.arange(T, dtype=int)
        z = np.empty(T, dtype=float)
        k = np.empty(T, dtype=float)
        k_next = np.empty(T, dtype=float)
        i = np.empty(T, dtype=float)
        d = np.empty(T, dtype=float)
        b = np.empty(T, dtype=float)
        b_next = np.empty(T, dtype=float)

        z[0], k[0], b[0] = float(z0), float(k0), float(b0)

        for tt in range(T):
            kp, bp = self.policy(z[tt], k[tt], b[tt], t=tt)
            k_next[tt], b_next[tt] = kp, bp
            i[tt] = kp - (1.0 - self.p.DELTA) * k[tt]
            d[tt] = env.dividend(z[tt], k[tt], i[tt], b[tt], bp)

            if tt < T - 1:
                z[tt+1], k[tt+1], b[tt+1] = env.transition(z[tt], kp, bp, custom_rng=rng)

        return SimResult(t=t, z=z, k=k, k_next=k_next, i=i, d=d, b=b, b_next=b_next)

    def simulate_irf(
        self,
        env: InvestmentEnvironment | None = None,
        T: int = 40,
        shock_size_log: float = 0.05,
        shock_time: int = 0,
        z0: float = 1.0,
        k0: float | None = None,
        b0: float | None = None,
        deterministic_after: bool = True,
        seed: int = 0,
    ) -> dict:
        """Impulse response function. Uses the agent's own environment if env is not provided."""
        env = env or self.env
        p = self.p

        if k0 is None:
            k0 = float(self.fixed_point())
        if b0 is None:
            b0 = p.THETA * k0

        tgrid = np.arange(T, dtype=int)
        rng_base = np.random.default_rng(seed)
        rng_shk  = np.random.default_rng(seed)

        def step_z(z_now: float, rng) -> float:
            if p.SIGMA_EPS <= 0:
                return z_now
            if deterministic_after:
                return float(np.exp(p.RHO * np.log(max(z_now, 1e-12))))
            return float(np.exp(p.RHO * np.log(max(z_now, 1e-12)) + rng.normal(0.0, p.SIGMA_EPS)))

        def run_path(apply_shock: bool, rng) -> SimResult:
            z = np.empty(T, dtype=float)
            k = np.empty(T, dtype=float)
            k_next = np.empty(T, dtype=float)
            i_arr = np.empty(T, dtype=float)
            d = np.empty(T, dtype=float)
            b = np.empty(T, dtype=float)
            b_next = np.empty(T, dtype=float)

            z[0], k[0], b[0] = float(z0), float(k0), float(b0)

            if apply_shock and shock_time == 0:
                z[0] = float(np.exp(np.log(max(z[0], 1e-12)) + shock_size_log))

            for tt in range(T):
                kp, bp = self.policy(z[tt], k[tt], b[tt], t=tt)
                k_next[tt], b_next[tt] = kp, bp
                i_arr[tt] = kp - (1.0 - p.DELTA) * k[tt]
                d[tt] = float(env.dividend(z[tt], k[tt], i_arr[tt], b[tt], bp))

                if tt < T - 1:
                    k[tt+1] = max(kp, 1e-8)
                    b[tt+1] = float(bp)
                    z[tt+1] = step_z(z[tt], rng)
                    if apply_shock and (tt + 1) == shock_time:
                        z[tt+1] = float(np.exp(np.log(max(z[tt+1], 1e-12)) + shock_size_log))

            return SimResult(t=tgrid, z=z, k=k, k_next=k_next, i=i_arr, d=d, b=b, b_next=b_next)

        base = run_path(False, rng_base)
        shk  = run_path(True,  rng_shk)

        diff = SimResult(
            t=tgrid,
            z=shk.z - base.z, k=shk.k - base.k,
            k_next=shk.k_next - base.k_next, i=shk.i - base.i,
            d=shk.d - base.d, b=shk.b - base.b, b_next=shk.b_next - base.b_next,
        )
        pct = {
            "z": 100.0 * (shk.z / base.z - 1.0),
            "k": 100.0 * (shk.k / base.k - 1.0),
            "i": 100.0 * (shk.i / base.i - 1.0),
            "d": 100.0 * (shk.d / base.d - 1.0),
            "b": 100.0 * (shk.b / np.maximum(base.b, 1e-12) - 1.0),
        }
        return {"t": tgrid, "base": base, "shk": shk, "diff": diff, "pct": pct}

    def policy_curve(self, z_grid: np.ndarray, k_grid: np.ndarray, z_idx: int = 0, b: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (k_grid, kprime_grid) for plotting at a given z index and debt b."""
        z_val = float(z_grid[z_idx])
        kprime_vals = np.array([self.policy(z_val, float(kv), b)[0] for kv in k_grid], dtype=float)
        return k_grid, kprime_vals


class RationalInvestmentAgent(InvestmentAgent):

    def __init__(self, env: InvestmentEnvironment):
        super().__init__(env, name="Rational (VFI)")
        self.initialize_value_function()

    def initialize_value_function(self):
        self.v = np.zeros((self.env.actual_nz, self.env.p.N_k))
        self.policy_k = np.zeros((self.env.actual_nz, self.env.p.N_k))
        for i_z in range(self.env.actual_nz):
            self.v[i_z, :] = self.env.production(self.env.z_grid[i_z], self.env.k_grid) / self.env.p.R

    def _vfi_dividend(self, k_next, k_current, z):
        """VFI objective dividend for W(z,k) = V(z,k,0).

        Since V(z,k,b) = W(z,k) - (1+R)*b, the VFI objective at b=0 is
        exactly the GP observation target: f(z,k) - i - psi(i,k) + b_next*(1 - beta*(1+R)).
        """
        i = k_next - (1 - self.env.p.DELTA) * k_current
        b_next = self.env.optimal_b_next(k_next)
        return self.env.gp_observation(z, k_current, i, b_next)

    def fit(self, tol=1e-5, max_iter=500):
        for _ in range(max_iter):
            v_new = np.zeros_like(self.v)
            policy_new = np.zeros_like(self.policy_k)

            v_expected = self.env.P @ self.v
            ev_funcs = [
                interp1d(self.env.k_grid, v_expected[i, :], kind="linear", fill_value="extrapolate") # type: ignore[reportUnknownMemberType]
                for i in range(self.env.actual_nz)
            ]

            for i_z in range(self.env.actual_nz):
                z = self.env.z_grid[i_z]
                cont_func = ev_funcs[i_z]

                for i_k, k in enumerate(self.env.k_grid):
                    def objective(k_prime, _k=k, _z=z):
                        return -(self._vfi_dividend(k_prime, _k, _z) + self.env.p.BETA * cont_func(k_prime))

                    res = minimize_scalar(objective, bounds=(self.env.p.K_min, self.env.p.K_max), method="bounded")
                    v_new[i_z, i_k] = -res.fun
                    policy_new[i_z, i_k] = res.x

            if np.max(np.abs(v_new - self.v)) < tol:
                break

            self.v = v_new
            self.policy_k = policy_new

        self._pol_funcs = [
            interp1d(self.env.k_grid, self.policy_k[i, :], kind="linear", fill_value="extrapolate") # type: ignore[reportUnknownMemberType]
            for i in range(self.env.actual_nz)
        ]
        return self

    def _z_to_idx(self, z):
        return int(np.argmin(np.abs(self.env.z_grid - z)))

    def policy(self, z: float, k: float, b: float = 0.0, t=None) -> tuple[float, float]:
        z_idx = self._z_to_idx(z)
        k_next = float(self._pol_funcs[z_idx](k))
        return k_next, self.env.optimal_b_next(k_next)

    def fixed_point(self) -> float:
        def objective(k):
            return self.policy(1.0, k)[0] - k
        return brentq(objective, self.env.k_grid[0], self.env.k_grid[-1]) # pyright: ignore[reportReturnType]


@dataclass
class InvestmentAgentParameters:
    H: float = 0.05  # exploration regularization parameter\
    KAPPA_R: float = 0.01


def calibrated_entropy_regularization_parameter(env: InvestmentEnvironment, gp_params: GPBeliefParameters) -> float:
    """Calibrates h from environment primitives so the initial entropy target
    h * tr(Sigma_0) = 0.5 * ln(N_k), where tr(Sigma_0) = N_k * sigma0^2.
    
        h = 0.5 * ln(N_k) / (N_k * sigma0^2)
    """
    N = env.p.N_k
    return 0.5 * np.log(N) / (N * gp_params.kernel.sigma0_sq)

class ExperienceReasoningAgent(InvestmentAgent):
    def __init__(self, env, gp, agent_params, experience_only=False, seed=42):
        super().__init__(env, name="Experience-Reasoning Agent")
        self.gp = gp
        self.agent_params = agent_params
        self.experience_only = experience_only
        self.rng = np.random.default_rng(seed)

    def _get_action_queries(self, z, k, b=0.0):
        k_safe = max(k, 1e-8)
        i_candidates = self.env.k_grid - (1.0 - self.env.p.DELTA) * k_safe
        b_next_candidates = np.array([self.env.optimal_b_next(kp) for kp in self.env.k_grid])
        dividends = np.array([
            self.env.dividend(z, k_safe, i, b, bn)
            for i, bn in zip(i_candidates, b_next_candidates)
        ])

        valid_mask = dividends >= -1e-8
        if not np.any(valid_mask):
            best_idx = np.argmax(dividends)
            valid_mask[best_idx] = True

        valid_k = self.env.k_grid[valid_mask]
        valid_i = i_candidates[valid_mask]
        X_query = np.column_stack([
            np.full(len(valid_k), z),
            np.full(len(valid_k), k_safe),
            valid_i
        ])
        return valid_k, valid_i, X_query

    def _entropy_policy(self, q, std):
        N = len(q)
        H_max = np.log(N)
        lower_bound_entropy = self.agent_params.H * np.sum(std**2)

        if lower_bound_entropy <= 1e-8:
            p = np.zeros(N)
            p[np.argmax(q)] = 1.0
            return p, 0.0

        if lower_bound_entropy >= H_max - 1e-8:
            return np.ones(N) / N, np.inf

        def f(delta):
            logits = q / max(delta, 1e-12)
            logits -= np.max(logits)
            p = np.exp(logits)
            p /= np.sum(p)
            return -np.sum(p * np.log(p + 1e-12)) - lower_bound_entropy

        try:
            delta_star = brentq(f, 1e-6, 1e6, maxiter=200)
        except ValueError:
            delta_star = 1e-6 if abs(f(1e-6)) < abs(f(1e6)) else 1e6

        logits = q / max(delta_star, 1e-12)
        logits -= np.max(logits)
        p = np.exp(logits)
        p /= np.sum(p)
        return p, float(delta_star)  # type: ignore

    def get_beliefs(self, z, k, b=0.0):
        k_cands, _, X_q = self._get_action_queries(z, k, b)
        mean, std = self.gp.predict(X_q, return_std=True)
        return k_cands, X_q, mean, std

    def reason(self, X_q, delta_E, std_E):
        kappa = self.agent_params.KAPPA_R
        if self.experience_only or kappa <= 0 or delta_E <= 0 or self.agent_params.H <= 0:
            return std_E

        _, Sigma_E = self.gp.predict_full(X_q)
        vals_E, V = np.linalg.eigh(Sigma_E)
        water_level = kappa / (self.agent_params.H * delta_E)
        vals_R = np.minimum(vals_E, water_level)
        Sigma_R = V @ np.diag(vals_R) @ V.T
        return np.sqrt(np.maximum(np.diag(Sigma_R), 0))

    def policy(self, z, k, b=0.0, t=None):
        k_cands, X_q, mean, std_E = self.get_beliefs(z, k, b)
        _, delta_E = self._entropy_policy(mean, std_E)
        std_R = self.reason(X_q, delta_E, std_E)
        probs_R, _ = self._entropy_policy(mean, std_R)

        chosen_idx = self.rng.choice(len(k_cands), p=probs_R)
        k_next = float(k_cands[chosen_idx])
        b_next = self.env.optimal_b_next(k_next)

        return k_next, b_next

    def get_greedy_action(self, z, k, b=0.0):
        k_cands, _, X_q = self._get_action_queries(z, k, b)
        mean = self.gp.predict(X_q, return_std=False)
        k_next = float(k_cands[np.argmax(mean)])
        return k_next, self.env.optimal_b_next(k_next)

    def get_expected_action(self, z, k, b=0.0):
        k_cands, X_q, mean, std_E = self.get_beliefs(z, k, b)
        _, delta_E = self._entropy_policy(mean, std_E)
        std_R = self.reason(X_q, delta_E, std_E)
        probs_R, _ = self._entropy_policy(mean, std_R)
        k_next = float(np.dot(probs_R, k_cands))
        return k_next, self.env.optimal_b_next(k_next)

    def fixed_point(self):
        diffs = [abs(self.get_greedy_action(1.0, k)[0] - k) for k in self.env.k_grid]
        return float(self.env.k_grid[np.argmin(diffs)])