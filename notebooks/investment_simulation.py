from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, brentq
from scipy.spatial.distance import cdist
from scipy.special import softmax
import quantecon as qe
# ==========================================
# 1. PARAMETERS DATACLASS
# ==========================================
@dataclass
class InvestmentParameters:
    # Economic Parameters
    ALPHA: float = 0.33
    DELTA: float = 0.10
    R: float = 0.04
    
    # These will be set based on the scenario
    KAPPA: float = 0.0      # Adjustment cost parameter
    RHO: float = 0.9        # Persistence
    SIGMA_EPS: float = 0.0  # Volatility (0 = Deterministic)

    # Grid Parameters
    N_k: int = 100
    N_z: int = 5            # Number of shock states (if stochastic)
    K_min: float = 0.1
    K_max: float = 20.0

    # Calculated post-init
    BETA: float = field(init=False)

    def __post_init__(self):
        self.BETA = 1 / (1 + self.R)

# ==========================================
# 2. SIMULATION MODEL CLASS
# ==========================================
class RationalInvestmentSimulation:
    def __init__(self, params: InvestmentParameters):
        self.p = params
        self.setup_grids()
        self.initialize_value_function()

    def setup_grids(self):
        """
        Sets up z_grid (shocks) and k_grid (capital).
        Handles the 'No Shock' case by forcing a 1-point grid.
        """
        # --- Handle Shocks ---
        if self.p.SIGMA_EPS > 0 and self.p.N_z > 1:
            # Stochastic Case: Use Tauchen
            mc = qe.markov.approximation.tauchen(
                self.p.N_z, self.p.RHO, self.p.SIGMA_EPS, mu=0, n_std=3
            )
            self.z_grid = np.exp(mc.state_values)
            self.P = mc.P
            self.actual_nz = self.p.N_z
        else:
            # Deterministic Case (No shocks)
            self.z_grid = np.array([1.0])
            self.P = np.array([[1.0]])
            self.actual_nz = 1  # Force N_z to 1 internally

        # --- Handle Capital ---
        self.k_grid = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)

    def initialize_value_function(self):
        """Initial guess: Value if we just maintained capital forever."""
        self.v = np.zeros((self.actual_nz, self.p.N_k))
        self.policy_k = np.zeros((self.actual_nz, self.p.N_k))

        # Simple guess: V = Output / Interest Rate
        for i_z in range(self.actual_nz):
            output = self.z_grid[i_z] * (self.k_grid ** self.p.ALPHA)
            self.v[i_z, :] = output / self.p.R

    def get_dividend(self, k_next, k_current, z):
        """Calculates: Output - Investment - Adjustment Costs"""
        investment = k_next - (1 - self.p.DELTA) * k_current
        
        # Adjustment Costs: (kappa/2) * (I/K)^2 * K
        adj_cost = 0.0
        if self.p.KAPPA > 0:
            adj_cost = (self.p.KAPPA / 2) * (investment**2 / k_current)
            
        return z * (k_current ** self.p.ALPHA) - investment - adj_cost

    def solve(self, tol=1e-5, max_iter=500, verbose=False):
        """Standard Value Function Iteration"""
        
        for it in range(max_iter):
            v_new = np.zeros_like(self.v)
            policy_new = np.zeros_like(self.policy_k)

            # 1. Compute Expected Value E[V(k', z') | z]
            # Matrix multiplication: (N_z x N_z) @ (N_z x N_k) -> (N_z x N_k)
            v_expected = self.P @ self.v

            # 2. Interpolate Expected Value for off-grid optimization
            ev_funcs = [
                interp1d(self.k_grid, v_expected[i, :], kind='linear', fill_value="extrapolate")
                for i in range(self.actual_nz)
            ]

            # 3. Optimize for every state (z, k)
            for i_z in range(self.actual_nz):
                z = self.z_grid[i_z]
                cont_func = ev_funcs[i_z]

                for i_k, k in enumerate(self.k_grid):
                    
                    # Objective: maximize Dividend + Beta * EV
                    def objective(k_prime):
                        div = self.get_dividend(k_prime, k, z)
                        return -(div + self.p.BETA * cont_func(k_prime))

                    # Bounded search to ensure we stay within grid limits
                    res = minimize_scalar(
                        objective, 
                        bounds=(self.p.K_min, self.p.K_max), 
                        method='bounded'
                    )
                    
                    v_new[i_z, i_k] = -res.fun
                    policy_new[i_z, i_k] = res.x

            diff = np.max(np.abs(v_new - self.v))
            self.v = v_new
            self.policy_k = policy_new

            if diff < tol:
                if verbose: print(f"Converged at iteration {it}")
                break
        else:
            print("Warning: Max iterations reached without convergence.")

    def simulate(self, T=60):
        """Simulates a path of length T."""
        
        # 1. Start at Steady State approx (Deterministic SS)
        k_ss_approx = ((self.p.ALPHA / (self.p.R + self.p.DELTA))**(1/(1-self.p.ALPHA)))
        
        k_path = np.zeros(T)
        k_path[0] = k_ss_approx
        
        # Initialize shock index (start at mean)
        z_idx = self.actual_nz // 2
        
        # Interpolators for policy functions
        pol_funcs = [
            interp1d(self.k_grid, self.policy_k[i, :], kind='linear', fill_value="extrapolate")
            for i in range(self.actual_nz)
        ]

        # For consistent random shocks
        rng = np.random.default_rng(42)

        for t in range(T - 1):
            # A. Determine Next Capital
            k_path[t+1] = pol_funcs[z_idx](k_path[t])
            
            # B. Determine Next Shock
            if self.actual_nz > 1:
                z_idx = rng.choice(self.actual_nz, p=self.P[z_idx, :])
            # If actual_nz == 1, z_idx stays 0 (no shock)

        return k_path
    

class InvestmentEnvironment:
    def __init__(self, params: InvestmentParameters):
        self.p = params
        
    def production(self, z, k):
        """Standard Cobb-Douglas production."""
        return z * (k ** self.p.ALPHA)
    
    def dividend(self, z, k, i):
        """Calculates dividend given investment i."""
        adj_cost = 0.0
        if self.p.KAPPA > 0:
            adj_cost = (self.p.KAPPA / 2) * (i**2 / k)
        
        return self.production(z, k) - i - adj_cost

    def get_reward(self, z, k, k_prime):
        """
        Calculates the dividend (Reward).
        Action is k_prime (where we want to be next period).
        """
        gross_investment = k_prime - (1 - self.p.DELTA) * k
    
        k_safe = max(k, 1e-8)
        
        return self.dividend(z, k_safe, gross_investment)

    def transition(self, z, k_prime):
        """
        Moves the economy to the next state.
        Returns: (z_next, k_next)
        """
        # 1. Update Shock (AR(1) in logs)
        if self.p.SIGMA_EPS > 0:
            eps = np.random.normal(0, self.p.SIGMA_EPS)
            # log(z') = rho * log(z) + eps  =>  z' = z^rho * exp(eps)
            z_next = np.exp(self.p.RHO * np.log(z) + eps)
        else:
            z_next = z

        k_next = k_prime
        
        # (Optional) Enforce natural bounds (can't have negative capital)
        k_next = max(k_next, 1e-5) 

        return z_next, k_next
    

def rbf_kernel_generator(sigma=1.0, length_scales=np.array([1.0, 1.0, 1.0])):
    """
    sigma: Vertical amplitude of the belief function.
    length_scales: A vector [l_z, l_k, l_kp]
    """
    # Pre-convert to array for division later
    l_scales = np.array(length_scales)
    
    def rbf_kernel(X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # NORMALIZE the data by the length scales
        X_scaled = X / l_scales
            
        # Compute Squared Euclidean Distance on the normalized points
        dists = cdist(X_scaled, X_scaled, metric='sqeuclidean')
        
        # Standard RBF formula
        # Using 0.5 to match the common 1/(2*l^2) definition
        K = sigma**2 * np.exp(-0.5 * dists)
        
        return K

    return rbf_kernel

@dataclass
class ReasoningParameters:
    KAPPA: float = 0.01  # Cost of reasoning
    H_MIN: float = 0.05   # Minimum entropy scaling for decision-making


class InvestmentReasoningAgent:
    def __init__(self, params, reasoning_params, kernel=None):
        self.p = params
        self.rp = reasoning_params
        self.setup_grids()
        
        # Initialize the Kernel Function
        if kernel is None:
            # Heuristic length scales: 0.5 * range of each dimension
            z_range = np.max(self.z_grid) - np.min(self.z_grid)
            k_range = np.max(self.k_grid) - np.min(self.k_grid)

            l_z = 0.5 * z_range if z_range > 0 else 1.0
            l_k = 0.5 * k_range
            l_kp = l_k
            self.kernel = rbf_kernel_generator(sigma=1.0, length_scales=np.array([l_z, l_k, l_kp]))
        else:
            self.kernel = kernel

        # Design matrix of all (z, k, k') combinations
        self.points, self.z_mesh, self.k_mesh, self.kp_mesh = self._build_design_matrix()
        self.N_total = self.points.shape[0]
        
        # Belief initialization (Q-Function Tensor)
        self.mu_tensor = np.zeros((self.actual_nz, self.p.N_k, self.p.N_k))
        
        # The Covariance Matrix (Big N x N object)
        self.sigma_matrix = self._compute_full_kernel()

    def setup_grids(self):
        """
        Sets up z_grid (shocks) and k_grid (capital).
        Handles the 'No Shock' case by forcing a 1-point grid.
        """
        # --- Handle Shocks ---
        if self.p.SIGMA_EPS > 0 and self.p.N_z > 1:
            # Stochastic Case: Use Tauchen
            mc = qe.markov.approximation.tauchen(
                self.p.N_z, self.p.RHO, self.p.SIGMA_EPS, mu=0, n_std=3
            )
            self.z_grid = np.exp(mc.state_values)
            self.P = mc.P
            self.actual_nz = self.p.N_z
        else:
            # Deterministic Case
            self.z_grid = np.array([1.0])
            self.P = np.array([[1.0]])
            self.actual_nz = 1

        self.k_grid = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)
        
    def _build_design_matrix(self):
        """
        Creates the flattened (N, 3) matrix of all state-action pairs.
        Returns: points, Z_mesh, K_mesh, Kp_mesh
        """
        # Create 3D Meshgrid
        # Indexing 'ij' ensures: Z varies slowest, then K, then K' (Action)
        Z, K, Kp = np.meshgrid(self.z_grid, self.k_grid, self.k_grid, indexing='ij')
        
        # Flatten into design matrix
        # Columns: [Productivity, Current Capital, Next Capital]
        points = np.column_stack([Z.ravel(), K.ravel(), Kp.ravel()])
        
        return points, Z, K, Kp

    def _compute_full_kernel(self):
        """
        Calculates the full covariance matrix and adds numerical jitter.
        """
        K_matrix = self.kernel(self.points)
        
        # add jitter for numerical stability
        K_matrix += np.eye(self.N_total) * 1e-6
        
        return K_matrix

    @property
    def mu_flat(self):
        """Returns the flat view of the mean (for math)."""
        return self.mu_tensor.ravel()
    
    @mu_flat.setter
    def mu_flat(self, value):
        """Sets the tensor from a flat vector."""
        self.mu_tensor = value.reshape(self.actual_nz, self.p.N_k, self.p.N_k)

    def get_indices_for_state(self, z_idx, k_idx):
        """
        Returns the range of indices in the flat matrix 
        corresponding to ALL actions at state (z, k).
        Used to slice the Covariance Matrix for reasoning.
        """
        # Because we used meshgrid(indexing='ij'), the memory is contiguous 
        # for the last dimension (Kp).
        
        # How many points in one Z-slice? (N_k * N_k)
        z_stride = self.p.N_k * self.p.N_k
        
        # How many points in one K-slice? (N_k)
        k_stride = self.p.N_k
        
        start_idx = (z_idx * z_stride) + (k_idx * k_stride)
        end_idx = start_idx + k_stride
        
        return start_idx, end_idx
    
    def optimal_policy(self, z_idx, k_idx):
        """
        Proposition: Optimal Action Policy
        Solves for the shadow price (delta) that satisfies the entropy constraint.
        """
        q_vals = self.mu_tensor[z_idx, k_idx, :]
    
        # Calculate target entropy: h * sum of posterior uncertainties
        start, end = self.get_indices_for_state(z_idx, k_idx)
        uncertainties = np.sqrt(np.diag(self.sigma_matrix)[start:end])
        target_entropy = self.rp.H_MIN * np.sum(uncertainties)

        # If uncertainty is effectively zero, delta is zero (Greedy)
        if target_entropy < 1e-7:
            return np.eye(len(q_vals))[np.argmax(q_vals)], 0.0

        # Define the root-finding objective: Actual Entropy - Target Entropy
        def objective(delta):
            probs = softmax(q_vals / delta)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return entropy - target_entropy

        # Solve for delta. 
        try:
            # Explicitly cast to float to satisfy type checkers
            delta_star = float(brentq(objective, 1e-6, 100.0)) # type: ignore
        except ValueError:
            # If target_entropy is too high to reach even at delta=100
            delta_star = 100.0 if objective(100.0) < 0 else 1e-6

        # Safety: Ensure delta is not exactly zero to avoid Inf in division
        delta_star = max(delta_star, 1e-8)
        
        # Calculate probabilities with the discovered shadow price
        probs = softmax(q_vals / delta_star)
        
        return probs, delta_star


import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq
from scipy.special import softmax
from scipy.spatial.distance import cdist
import quantecon as qe

# ============================================================
# 0) Kernel
# ============================================================
def rbf_kernel_generator(sigma=1.0, length_scales=np.array([1.0, 1.0, 1.0])):
    l_scales = np.array(length_scales, dtype=float)

    def rbf_kernel(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = X / l_scales
        dists = cdist(X_scaled, X_scaled, metric="sqeuclidean")
        return (sigma**2) * np.exp(-0.5 * dists)

    return rbf_kernel


# ============================================================
# 1) GP-on-a-grid belief: (mu, Sigma) + linear-Gaussian updates
# ============================================================
class GridGPBelief:
    """
    GP belief over a finite design set (your full (z,k,k') grid).
    Stores mean mu (N,) and covariance Sigma (N,N).

    Provides Kalman-style conditioning for linear observations:
      y = H mu + noise,  H is implemented sparsely via (idx, w) pairs.

    This matches the GP closure-under-conditioning formulas, but avoids recomputing kernels.
    """

    def __init__(self, points: np.ndarray, kernel_fn, jitter=1e-6, mu0=None):
        self.points = np.asarray(points, dtype=float)
        self.N = self.points.shape[0]
        self.kernel_fn = kernel_fn

        self.Sigma = self.kernel_fn(self.points)
        self.Sigma += np.eye(self.N) * float(jitter)

        if mu0 is None:
            self.mu = np.zeros(self.N, dtype=float)
        else:
            mu0 = np.asarray(mu0, dtype=float)
            assert mu0.shape == (self.N,)
            self.mu = mu0.copy()

    def condition_sparse(self, idx: np.ndarray, w: np.ndarray, y: float, noise_var: float):
        """
        Condition on a single scalar linear observation:
          y = sum_j w_j * f[idx_j] + eps,  eps ~ N(0, noise_var)

        idx: (m,) integer indices into latent vector
        w:   (m,) weights
        """
        idx = np.asarray(idx, dtype=int)
        w = np.asarray(w, dtype=float)

        mu = self.mu
        Sigma = self.Sigma

        # innovation
        y_hat = float(w @ mu[idx])
        innov = float(y - y_hat)

        # S = w' Sigma_ii w + noise
        Sigma_ii = Sigma[np.ix_(idx, idx)]
        S = float(w @ Sigma_ii @ w) + float(noise_var)

        # K = Sigma H' / S where H has nonzeros w at idx
        Sigma_Ht = Sigma[:, idx] @ w  # (N,)
        K = Sigma_Ht / S              # (N,)

        # posterior
        self.mu = mu + K * innov
        self.Sigma = Sigma - np.outer(K, Sigma_Ht)

    def marginal_std(self):
        return np.sqrt(np.clip(np.diag(self.Sigma), 0.0, np.inf))


# ============================================================
# 2) Investment environment (your primitives)
# ============================================================
@dataclass
class InvestmentParameters:
    ALPHA: float = 0.33
    DELTA: float = 0.10
    R: float = 0.04
    RHO: float = 0.9
    SIGMA_EPS: float = 0.02

    N_k: int = 80
    N_z: int = 5
    K_min: float = 0.1
    K_max: float = 20.0

    @property
    def BETA(self):
        return 1.0 / (1.0 + self.R)


class InvestmentEnvironment:
    def __init__(self, params: InvestmentParameters):
        self.p = params
        self.rng = np.random.default_rng(0)

    def production(self, z, k):
        return z * (k ** self.p.ALPHA)

    def dividend(self, z, k, i):
        # linear utility base
        return self.production(z, k) - i

    def transition(self, z, k, i):
        k_next = (1.0 - self.p.DELTA) * k + i
        k_next = max(k_next, 1e-8)

        if self.p.SIGMA_EPS > 0:
            eps = self.rng.normal(0.0, self.p.SIGMA_EPS)
            z_next = np.exp(self.p.RHO * np.log(z) + eps)
        else:
            z_next = z

        return z_next, k_next


# ============================================================
# 3) Reasoning + Experience agent
# ============================================================
@dataclass
class ReasoningParameters:
    KAPPA: float = 0.01   # reasoning cost
    H_MIN: float = 0.05   # entropy floor scaling
    SIGMA_E2: float = 1e-3  # noise variance for experience TD signal
    SIGMA_R2_BASE: float = 1e-2  # baseline reasoning signal noise (scaled by allocation)


class InvestmentReasoningAgent:
    """
    Discrete GP over Q(z,k,k').
    Period cycle:
      1) Experience update (one TD-style scalar observation)
      2) Compute draft policy, delta^E
      3) Reasoning allocation (reverse water filling on action-slice covariance)
      4) Reasoning update (multi-signal linear observation at current state)
      5) Choose action from reasoned policy
    """

    def __init__(self, p: InvestmentParameters, rp: ReasoningParameters, kernel_fn=None):
        self.p = p
        self.rp = rp

        self._setup_grids()

        if kernel_fn is None:
            z_range = float(np.max(self.z_grid) - np.min(self.z_grid))
            k_range = float(np.max(self.k_grid) - np.min(self.k_grid))
            l_z = 0.5 * z_range if z_range > 0 else 1.0
            l_k = 0.5 * k_range if k_range > 0 else 1.0
            l_kp = l_k
            kernel_fn = rbf_kernel_generator(
                sigma=1.0, length_scales=np.array([l_z, l_k, l_kp], dtype=float)
            )

        self.points, self.Zm, self.Km, self.Kpm = self._build_design_matrix()
        self.gp = GridGPBelief(self.points, kernel_fn, jitter=1e-6)

    # -------------------- grids / indexing --------------------
    def _setup_grids(self):
        if self.p.SIGMA_EPS > 0 and self.p.N_z > 1:
            mc = qe.markov.approximation.tauchen(
                self.p.N_z, self.p.RHO, self.p.SIGMA_EPS, mu=0, n_std=3
            )
            self.z_grid = np.exp(mc.state_values)
            self.P = mc.P
            self.Nz = self.p.N_z
        else:
            self.z_grid = np.array([1.0])
            self.P = np.array([[1.0]])
            self.Nz = 1

        self.k_grid = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)
        self.Nk = self.p.N_k

    def _build_design_matrix(self):
        Z, K, Kp = np.meshgrid(self.z_grid, self.k_grid, self.k_grid, indexing="ij")
        points = np.column_stack([Z.ravel(), K.ravel(), Kp.ravel()])
        return points, Z, K, Kp

    def flat_index(self, z_idx, k_idx, kp_idx):
        # ordering matches meshgrid(indexing='ij') and ravel()
        return z_idx * (self.Nk * self.Nk) + k_idx * self.Nk + kp_idx

    def state_action_slice(self, z_idx, k_idx):
        start = z_idx * (self.Nk * self.Nk) + k_idx * self.Nk
        end = start + self.Nk
        return start, end  # indices for all actions kp at that state

    def nearest_index_1d(self, grid, x):
        return int(np.clip(np.searchsorted(grid, x), 0, len(grid) - 1))

    # -------------------- policy delta (entropy constraint) --------------------
    def policy_from_mu(self, z_idx, k_idx):
        """
        Solve for delta (temperature) that satisfies:
          H(pi) >= h * sum sigma(action)
        with pi(a) ∝ exp(Q(a)/delta).
        """
        start, end = self.state_action_slice(z_idx, k_idx)
        q = self.gp.mu[start:end]
        std = np.sqrt(np.clip(np.diag(self.gp.Sigma)[start:end], 0.0, np.inf))

        target_entropy = float(self.rp.H_MIN * np.sum(std))
        if target_entropy < 1e-10:
            probs = np.zeros_like(q)
            probs[int(np.argmax(q))] = 1.0
            return probs, 0.0

        def ent_gap(delta):
            delta = max(float(delta), 1e-12)
            probs = softmax(q / delta)
            H = -np.sum(probs * np.log(probs + 1e-12))
            return float(H - target_entropy)

        try:
            delta_star = float(brentq(ent_gap, 1e-6, 200.0))
        except ValueError:
            # fallback: if constraint too tight/loose within bracket
            delta_star = 200.0 if ent_gap(200.0) < 0 else 1e-6

        delta_star = max(delta_star, 1e-12)
        probs = softmax(q / delta_star)
        return probs, delta_star

    # -------------------- Experience update --------------------
    def experience_update_td(self,
                             z_prev_idx, k_prev_idx, kp_prev_idx,
                             z_curr_idx, k_curr_idx):
        """
        Observe realized dividend u_t and arrived state; create TD-style linear observation:
          u = Q(prev) - beta * Q(curr, kp*_curr) + noise

        This is a single scalar update on the full GP.
        """
        beta = self.p.BETA

        # greedy action at arrived state under current mean (intermediate beliefs)
        start_c, end_c = self.state_action_slice(z_curr_idx, k_curr_idx)
        kp_star_curr = int(np.argmax(self.gp.mu[start_c:end_c]))

        i_prev = self.flat_index(z_prev_idx, k_prev_idx, kp_prev_idx)
        i_star = self.flat_index(z_curr_idx, k_curr_idx, kp_star_curr)

        idx = np.array([i_prev, i_star], dtype=int)
        w = np.array([1.0, -beta], dtype=float)

        return idx, w  # caller supplies y=u_observed

    # -------------------- Reasoning allocation + update --------------------
    def reasoning_update_reverse_waterfill(self, z_idx, k_idx, delta_E):
        """
        Implements your proposition on the action-slice covariance at current state.
        We set the target posterior eigenvalues:
          lambda_i^* = min(lambda_i^E, kappa / (h * delta_E))
        and then *simulate* reasoning as noisy observations in the eigenbasis.

        Mechanically: observe eta = V' Q_s + eps, choose eps variances so that posterior matches lambda^*.
        """
        start, end = self.state_action_slice(z_idx, k_idx)
        Sigma_s = self.gp.Sigma[start:end, start:end]
        mu_s = self.gp.mu[start:end]

        # if delta_E is ~0, reasoning is valueless under your structure; just skip
        if delta_E <= 1e-12:
            return

        h = float(self.rp.H_MIN)
        wlevel = float(self.rp.KAPPA / (h * delta_E))  # "water level" threshold

        # eigendecomposition of prior slice covariance
        evals, evecs = np.linalg.eigh(Sigma_s)
        evals = np.clip(evals, 0.0, np.inf)

        # target posterior eigenvalues
        evals_star = np.minimum(evals, wlevel)

        # implied signal noise variances per direction (from your proposition)
        # sigma_i^2 = (lambda^E lambda^*)/(lambda^E - lambda^*) for reduced directions, else inf (no signal)
        sig2 = np.full_like(evals, np.inf, dtype=float)
        mask = evals > evals_star + 1e-14
        sig2[mask] = (evals[mask] * evals_star[mask]) / (evals[mask] - evals_star[mask])

        # Build k signals: only where finite variance (we "reason" about those directions)
        idx_dirs = np.where(np.isfinite(sig2))[0]
        if idx_dirs.size == 0:
            return

        # Omega' = V' (restricted rows)
        # We create observations eta = V' Q_s + noise; set eta equal to its prior mean (certainty-equivalence)
        # so mean doesn't move on average; covariance shrinks as desired.
        Vt = evecs.T[idx_dirs, :]         # (k, Nk)
        eta_mean = Vt @ mu_s              # (k,)
        eta = eta_mean.copy()             # realized signals equal mean
        noise_vars = sig2[idx_dirs] + self.rp.SIGMA_R2_BASE  # small floor

        # Kalman update on the FULL GP for each signal row:
        # each signal is a linear combo over the slice entries only.
        # Implement each as a sparse linear observation over global indices.
        global_action_indices = np.arange(start, end, dtype=int)

        for r in range(Vt.shape[0]):
            w_local = Vt[r, :]  # weights over Nk actions
            # sparse representation over global latent vector: these Nk indices have weights w_local
            idx = global_action_indices
            w = w_local
            self.gp.condition_sparse(idx=idx, w=w, y=float(eta[r]), noise_var=float(noise_vars[r]))

    # -------------------- action choice --------------------
    def choose_action(self, z_idx, k_idx, use_reasoned=True):
        probs, delta = self.policy_from_mu(z_idx, k_idx)
        kp_idx = int(np.random.choice(self.Nk, p=probs))
        return kp_idx, probs, delta


# ============================================================
# 4) Example wiring: one period of the full learning cycle
# ============================================================
def one_step_cycle(env: InvestmentEnvironment,
                   agent: InvestmentReasoningAgent,
                   z, k,
                   z_idx, k_idx):
    """
    Illustrative loop for a single period.
    Uses the agent's current beliefs to choose k', observes dividend, updates beliefs.
    """

    # 1) pick action under current (reasoned) beliefs (initially same as prior)
    kp_idx, probs_R, delta_R = agent.choose_action(z_idx, k_idx)
    k_prime = agent.k_grid[kp_idx]

    # 2) execute in environment
    i = k_prime - (1.0 - agent.p.DELTA) * k
    u = env.dividend(z, k, i)
    z_next, k_next = env.transition(z, k, i)

    # 3) discretize next state to indices
    z_next_idx = agent.nearest_index_1d(agent.z_grid, z_next)
    k_next_idx = agent.nearest_index_1d(agent.k_grid, k_next)

    # 4) experience update (TD-style linear observation)
    idx, w = agent.experience_update_td(
        z_prev_idx=z_idx, k_prev_idx=k_idx, kp_prev_idx=kp_idx,
        z_curr_idx=z_next_idx, k_curr_idx=k_next_idx
    )
    agent.gp.condition_sparse(idx=idx, w=w, y=float(u), noise_var=float(agent.rp.SIGMA_E2))

    # 5) compute draft policy delta^E at arrived state (after experience update)
    _, delta_E = agent.policy_from_mu(z_next_idx, k_next_idx)

    # 6) reasoning update at arrived state using reverse water filling with delta^E
    agent.reasoning_update_reverse_waterfill(z_next_idx, k_next_idx, delta_E=delta_E)

    return (z_next, k_next, z_next_idx, k_next_idx, u, delta_E, delta_R)


# ------------------------------------------------------------
# Minimal run (deterministic or stochastic)
# ------------------------------------------------------------
p = InvestmentParameters(SIGMA_EPS=0.02, N_z=5, N_k=60)
rp = ReasoningParameters(KAPPA=0.01, H_MIN=0.05, SIGMA_E2=1e-3, SIGMA_R2_BASE=1e-2)

env = InvestmentEnvironment(p)
agent = InvestmentReasoningAgent(p, rp)

# initial state
z, k = 1.0, 10.0
z_idx = agent.nearest_index_1d(agent.z_grid, z)
k_idx = agent.nearest_index_1d(agent.k_grid, k)

T = 50
path = []
for t in range(T):
    z, k, z_idx, k_idx, u, delta_E, delta_R = one_step_cycle(env, agent, z, k, z_idx, k_idx)
    path.append((t, z, k, u, delta_E))


path
# path now contains simulated dynamics + learning statistics

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_diagnostics(agent, path, title_prefix=""):
    """
    path is list of tuples (t, z, k, u, delta_E) as in the example.
    agent is InvestmentReasoningAgent (contains grids and gp state).
    """

    t = np.array([row[0] for row in path], dtype=int)
    z = np.array([row[1] for row in path], dtype=float)
    k = np.array([row[2] for row in path], dtype=float)
    u = np.array([row[3] for row in path], dtype=float)
    delta_E = np.array([row[4] for row in path], dtype=float)

    # --- Plot 1: state path ---
    plt.figure()
    plt.plot(t, k)
    plt.xlabel("t")
    plt.ylabel("capital k_t")
    plt.title(f"{title_prefix}Capital path")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, z)
    plt.xlabel("t")
    plt.ylabel("productivity z_t")
    plt.title(f"{title_prefix}Productivity path")
    plt.tight_layout()

    # --- Plot 2: dividends and draft temperature ---
    plt.figure()
    plt.plot(t, u)
    plt.xlabel("t")
    plt.ylabel("dividend u_t")
    plt.title(f"{title_prefix}Dividend path")
    plt.tight_layout()

    plt.figure()
    plt.plot(t, delta_E)
    plt.xlabel("t")
    plt.ylabel("delta^E_t (entropy shadow price)")
    plt.title(f"{title_prefix}Draft policy temperature")
    plt.tight_layout()

    # --- Plot 3: snapshot of Q mean/uncertainty at a chosen time/state ---
    # choose last observed state from the path
    z_last, k_last = z[-1], k[-1]
    z_idx = agent.nearest_index_1d(agent.z_grid, z_last)
    k_idx = agent.nearest_index_1d(agent.k_grid, k_last)

    start, end = agent.state_action_slice(z_idx, k_idx)
    q_mean = agent.gp.mu[start:end]
    q_std = np.sqrt(np.clip(np.diag(agent.gp.Sigma)[start:end], 0.0, np.inf))

    kp_grid = agent.k_grid.copy()

    plt.figure()
    plt.plot(kp_grid, q_mean)
    plt.xlabel("next capital k'")
    plt.ylabel("posterior mean Q̂(z,k,k')")
    plt.title(f"{title_prefix}Posterior mean over actions at final state (z_idx={z_idx}, k_idx={k_idx})")
    plt.tight_layout()

    plt.figure()
    plt.plot(kp_grid, q_std)
    plt.xlabel("next capital k'")
    plt.ylabel("posterior std of Q")
    plt.title(f"{title_prefix}Posterior uncertainty over actions at final state (z_idx={z_idx}, k_idx={k_idx})")
    plt.tight_layout()

    # --- Plot 4: action distribution (policy) at final state ---
    probs, delta = agent.policy_from_mu(z_idx, k_idx)
    plt.figure()
    plt.plot(kp_grid, probs)
    plt.xlabel("next capital k'")
    plt.ylabel("pi(k' | z,k)")
    plt.title(f"{title_prefix}Policy over actions at final state (delta={delta:.3g})")
    plt.tight_layout()

    plt.show()


plot_learning_diagnostics(agent, path, title_prefix="GP TD + reasoning: ")

def plot_policy_function(agent, z_idx=0, use_expected=True):
    """
    Plots k -> k' policy for a fixed productivity index z_idx.
    
    use_expected=True  -> plots E[k' | policy]
    use_expected=False -> plots greedy argmax policy
    """

    k_grid = agent.k_grid
    Nk = agent.Nk

    k_next_vals = np.zeros(Nk)

    for k_idx in range(Nk):
        start, end = agent.state_action_slice(z_idx, k_idx)
        q_vals = agent.gp.mu[start:end]

        if use_expected:
            probs, _ = agent.policy_from_mu(z_idx, k_idx)
            k_next_vals[k_idx] = np.sum(probs * k_grid)
        else:
            kp_idx = int(np.argmax(q_vals))
            k_next_vals[k_idx] = k_grid[kp_idx]

    plt.figure()
    plt.plot(k_grid, k_next_vals)
    plt.plot(k_grid, k_grid, linestyle="--")  # 45-degree line
    plt.xlabel("Current capital k_t")
    plt.ylabel("Next capital k_{t+1}")
    plt.title(f"Policy function at z index {z_idx}")
    plt.tight_layout()
    plt.show()

plot_policy_function(agent, z_idx=0, use_expected=True)