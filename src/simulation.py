import numpy as np
from .investment import InvestmentAgent, InvestmentParameters
from .gaussian_process import GPBelief, GPBeliefParameters
from .investment import 
from scipy.optimize import brentq
from scipy.special import softmax

class ExperienceAgent(InvestmentAgent):
    def __init__(self, params: InvestmentParameters, gp: GPBelief, h_min: float = 0.05, seed: int = 0):
        super().__init__(params, name="Experience Agent")
        self.gp = gp
        self.h_min = h_min  # The 'h' parameter from your LaTeX
        self.rng = np.random.default_rng(seed)
        
        # Grid used purely to define the discrete action space C for the softmax
        self.k_candidates = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)

    def _get_action_queries(self, z: float, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper to generate the (N, 3) matrix of candidate actions for the GP."""
        i_candidates = self.k_candidates - (1.0 - self.p.DELTA) * k
        X_query = np.column_stack([
            np.full(self.p.N_k, z),
            np.full(self.p.N_k, k),
            i_candidates
        ])
        return self.k_candidates, i_candidates, X_query

    def _entropy_policy(self, q: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Implements Proposition 2: The endogenous temperature softmax.
        Returns the probability distribution AND the shadow price delta_t.
        """
        N = len(q)
        H_max = np.log(N)
        
        # The exploration constraint: h * sum(sigma)
        target_H = self.h_min * np.sum(std)

        # Base case 1: Uncertainty is basically zero -> Greedy Action
        if target_H <= 1e-8:
            p = np.zeros(N)
            p[np.argmax(q)] = 1.0
            return p, 0.0

        # Base case 2: Uncertainty is massive -> Uniform Random
        if target_H >= H_max - 1e-8:
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
            return H - target_H

        # Find the delta that satisfies the entropy constraint
        try:
            delta_star = brentq(f, 1e-6, 1e6, maxiter=200)
        except ValueError:
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
        k_cands, i_cands, X_query = self._get_action_queries(z, k)
        
        # GP Mental Simulation
        mean, std = self.gp.predict(X_query, return_std=True)
        
        # Calculate endogenous policy and temperature
        probs, delta_t = self._entropy_policy(mean, std)
        
        # Sample the action from the distribution
        chosen_idx = self.rng.choice(self.p.N_k, p=probs)
        chosen_kprime = float(k_cands[chosen_idx])
        
        if return_delta:
            return chosen_kprime, delta_t
        return chosen_kprime

    def get_greedy_action(self, z: float, k: float) -> float:
        """Used strictly for formulating the TD target (c-tilde in your definition)."""
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