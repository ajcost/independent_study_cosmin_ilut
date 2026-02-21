"""
GP-Learner vs Rational Agent: Firm Investment Problem
======================================================
Base case: linear utility u(d)=d, no idiosyncratic shocks.

Bellman:  v(k) = max_i { k^alpha - i + beta * v(k') }   s.t.  k' = (1-delta)*k + i

Rational agent: Euler eq  =>  1/beta = alpha*k'^(alpha-1) + (1-delta)
    Solved via cubic-spline VFI + Brent root-finding on the FOC.

GP learner: Prior GP centered above/below Q*, updates via experience-based
    TD learning. Policy extracted via quadratic interpolation around the
    discrete argmax for smooth curves.

Usage:  python gp_investment_sim.py
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# 1. PRIMITIVES
# ============================================================
alpha = 0.33
beta  = 0.96
delta = 0.10

k_star = (alpha / (1.0/beta - (1.0 - delta))) ** (1.0/(1.0 - alpha))
i_star = delta * k_star
print(f"Steady-state capital:    k* = {k_star:.4f}")
print(f"Steady-state investment: i* = {i_star:.4f}")

# ============================================================
# 2. RATIONAL BENCHMARK — smooth, continuous solution
# ============================================================
N_k = 500
k_lo, k_hi = 0.05 * k_star, 3.0 * k_star
k_grid = np.linspace(k_lo, k_hi, N_k)

def solve_rational(k_grid, tol=1e-10, max_iter=8000):
    """
    VFI on a fine grid, then cubic-spline V and recover policy via FOC.
    With linear utility the FOC is  beta * V'(k') = 1.
    """
    N = len(k_grid)
    V = np.zeros(N)

    N_i_inner = 3000  # very fine internal grid for VFI accuracy
    for it in range(max_iter):
        V_new = np.zeros(N)
        for j, k in enumerate(k_grid):
            output = k**alpha
            i_cands = np.linspace(0, output, N_i_inner)
            kp = (1 - delta)*k + i_cands
            Vp = np.interp(kp, k_grid, V, left=V[0], right=V[-1])
            obj = (output - i_cands) + beta * Vp
            V_new[j] = np.max(obj)
        if np.max(np.abs(V_new - V)) < tol:
            print(f"  VFI converged in {it+1} iterations")
            break
        V = V_new
    else:
        print("  VFI did not converge")

    # Cubic spline of converged V
    V_spl = CubicSpline(k_grid, V)

    # Smooth policy via FOC:  V'(k') = 1/beta
    target = 1.0 / beta
    i_pol  = np.zeros(N)
    kp_pol = np.zeros(N)
    for j, k in enumerate(k_grid):
        output = k**alpha
        kp_lo = (1 - delta)*k
        kp_hi = (1 - delta)*k + output

        d_lo = float(V_spl(kp_lo, 1))
        d_hi = float(V_spl(kp_hi, 1))

        if d_lo <= target:          # corner: i = 0
            kp_pol[j] = kp_lo
        elif d_hi >= target:        # corner: i = output
            kp_pol[j] = kp_hi
        else:                       # interior
            kp_pol[j] = brentq(lambda kp: float(V_spl(kp, 1)) - target,
                               kp_lo, kp_hi, xtol=1e-12)
        i_pol[j] = kp_pol[j] - (1 - delta)*k

    return V, V_spl, i_pol, kp_pol

print("Solving rational benchmark...")
V_rational, V_spl, i_rational, kprime_rational = solve_rational(k_grid)

# ============================================================
# 3. TRUE Q* ON A 2D GRID
# ============================================================
N_i_grid = 600
i_max = k_hi**alpha * 1.05
i_grid_g = np.linspace(0, i_max, N_i_grid)

def compute_Q_true(k_grid, i_grid, V_spl):
    Nk, Ni = len(k_grid), len(i_grid)
    Q = np.full((Nk, Ni), np.nan)
    for jk, k in enumerate(k_grid):
        output = k**alpha
        for ji, inv in enumerate(i_grid):
            if inv > output or inv < 0:
                continue
            kp = (1 - delta)*k + inv
            if kp < k_grid[0] or kp > k_grid[-1]:
                continue
            Q[jk, ji] = output - inv + beta * float(V_spl(kp))
    return Q

print("Computing Q* on 2-D grid...")
Q_true = compute_Q_true(k_grid, i_grid_g, V_spl)

# ============================================================
# 4. GP LEARNER (experience-based only)
# ============================================================
class GPLearner:
    def __init__(self, k_grid, i_grid, Q_true, bias,
                 sigma0=2.0, psi_k=1.0, psi_i=1.0, noise_var=0.01):
        self.kg = k_grid
        self.ig = i_grid
        self.s0 = sigma0
        self.pk = psi_k
        self.pi = psi_i
        self.nv = noise_var
        self.Qm = np.where(np.isnan(Q_true), np.nan, Q_true + bias)
        self.Qv = np.where(np.isnan(Q_true), 0.0, sigma0**2)

    # --- kernels (vectorised) ---
    def _Kgrid(self, k, i):
        dk = self.kg[:, None] - k
        di = self.ig[None, :] - i
        return self.s0**2 * np.exp(-self.pk*dk**2 - self.pi*di**2)

    def _Kpt(self, k1, i1, k2, i2):
        return self.s0**2 * np.exp(-self.pk*(k1-k2)**2 - self.pi*(i1-i2)**2)

    # --- smooth argmax via quadratic interpolation ---
    def _opt_i(self, jk):
        q = self.Qm[jk, :].copy()
        ok = ~np.isnan(q)
        if not np.any(ok):
            return self.ig[len(self.ig)//2]
        q_f = np.where(ok, q, -np.inf)
        j = int(np.argmax(q_f))
        if j == 0 or j == len(self.ig)-1:
            return self.ig[j]
        if np.isnan(q[j-1]) or np.isnan(q[j+1]):
            return self.ig[j]
        di = self.ig[1] - self.ig[0]
        ql, qm, qr = q[j-1], q[j], q[j+1]
        den = 2.0*(ql - 2*qm + qr)
        if abs(den) < 1e-15:
            return self.ig[j]
        shift = -di*(qr - ql)/den
        i_opt = self.ig[j] + np.clip(shift, -di, di)
        return float(np.clip(i_opt, 0, self.kg[jk]**alpha))

    def get_policy_kprime(self):
        kp = np.zeros(len(self.kg))
        for jk, k in enumerate(self.kg):
            kp[jk] = (1 - delta)*k + self._opt_i(jk)
        return kp

    # --- experience update ---
    def _Q_at(self, k, i):
        jk = np.argmin(np.abs(self.kg - k))
        return float(np.interp(i, self.ig,
                     np.nan_to_num(self.Qm[jk,:], nan=-1e10)))

    def experience_update(self, k, i, kp):
        div  = k**alpha - i
        jkp  = np.argmin(np.abs(self.kg - kp))
        i_nx = self._opt_i(jkp)
        Qnx  = self._Q_at(self.kg[jkp], i_nx)
        Qcur = self._Q_at(k, i)
        eta  = div + beta*Qnx
        td   = eta - Qcur

        Kd = self._Kgrid(k, i)
        Ko = self._Kgrid(self.kg[jkp], i_nx)
        cov_td = Kd - beta*Ko
        var_td = (self._Kpt(k,i,k,i)
                  + beta**2*self._Kpt(self.kg[jkp],i_nx,self.kg[jkp],i_nx)
                  - 2*beta*self._Kpt(k,i,self.kg[jkp],i_nx)
                  + self.nv)
        g = cov_td / var_td
        v = ~np.isnan(self.Qm)
        self.Qm[v] += (g*td)[v]
        self.Qv[v] -= ((cov_td**2)/var_td)[v]
        self.Qv = np.maximum(self.Qv, 0.0)
        return td

    def step(self, k_cur):
        jk = np.argmin(np.abs(self.kg - k_cur))
        k  = self.kg[jk]
        i  = self._opt_i(jk)
        i  = float(np.clip(i, 0, k**alpha))
        kp = (1-delta)*k + i
        td = self.experience_update(k, i, kp)
        return k, i, kp, td

# ============================================================
# 5. SIMULATION
# ============================================================
T  = 5
k0 = 0.5 * k_star
pk = 2.0 / k_star**2
pi = 2.0 / i_star**2
b_o, b_p = +3.0, -3.0

print(f"\nk0={k0:.4f}  k*={k_star:.4f}  psi_k={pk:.2f}  psi_i={pi:.2f}")

# Rational trajectory
k_rat = [k0]
for _ in range(T):
    ir = float(np.interp(k_rat[-1], k_grid, i_rational))
    k_rat.append((1-delta)*k_rat[-1] + ir)

# GP learners
go = GPLearner(k_grid, i_grid_g, Q_true, b_o, sigma0=2., psi_k=pk, psi_i=pi, noise_var=.1)
gp = GPLearner(k_grid, i_grid_g, Q_true, b_p, sigma0=2., psi_k=pk, psi_i=pi, noise_var=.1)

po = [go.get_policy_kprime()]; pp = [gp.get_policy_kprime()]
ko = [k0]; kp_path = [k0]
tdo = []; tdp = []

for t in range(T):
    _,_,kn,td = go.step(ko[-1]);  ko.append(kn); tdo.append(td); po.append(go.get_policy_kprime())
    _,_,kn,td = gp.step(kp_path[-1]); kp_path.append(kn); tdp.append(td); pp.append(gp.get_policy_kprime())

# ============================================================
# 6. PLOTTING
# ============================================================
fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    r"Firm Investment Policy: Rational vs GP Learner (Experience-Based Only)"
    "\n" + rf"$\alpha={alpha},\;\beta={beta},\;\delta={delta},\;k^*\!={k_star:.3f}$",
    fontsize=14, fontweight='bold', y=0.98)

gs = GridSpec(3,2,hspace=.35,wspace=.3,top=.92,bottom=.06)
cR, cB, blk = plt.cm.Reds, plt.cm.Blues, 'black'
sh = lambda t: 0.3+0.7*t/T

# A – optimistic
ax = fig.add_subplot(gs[0,0])
ax.plot(k_grid, kprime_rational, color=blk, lw=2.5, label=r'Rational $k_{t+1}^*$', zorder=10)
ax.plot(k_grid, k_grid, 'k--', lw=.8, alpha=.4, label=r'$45°$')
for t in range(T+1):
    ax.plot(k_grid, po[t], color=cR(sh(t)), lw=1.4, alpha=.85, label=f'$t={t}$')
ax.axvline(k_star, color='gray', ls=':', lw=.8, alpha=.5)
ax.set(xlabel=r'$k_t$', ylabel=r'$k_{t+1}$', xlim=(k_grid[0],k_grid[-1]), ylim=(k_grid[0],k_grid[-1]))
ax.set_title(r'Optimistic Prior ($\hat{Q}_0 = Q^* + b$)'); ax.legend(fontsize=7, loc='upper left', ncol=2)

# B – pessimistic
ax = fig.add_subplot(gs[0,1])
ax.plot(k_grid, kprime_rational, color=blk, lw=2.5, label=r'Rational $k_{t+1}^*$', zorder=10)
ax.plot(k_grid, k_grid, 'k--', lw=.8, alpha=.4, label=r'$45°$')
for t in range(T+1):
    ax.plot(k_grid, pp[t], color=cB(sh(t)), lw=1.4, alpha=.85, label=f'$t={t}$')
ax.axvline(k_star, color='gray', ls=':', lw=.8, alpha=.5)
ax.set(xlabel=r'$k_t$', ylabel=r'$k_{t+1}$', xlim=(k_grid[0],k_grid[-1]), ylim=(k_grid[0],k_grid[-1]))
ax.set_title(r'Pessimistic Prior ($\hat{Q}_0 = Q^* - |b|$)'); ax.legend(fontsize=7, loc='upper left', ncol=2)

# C – capital paths
ax = fig.add_subplot(gs[1,0])
tt = np.arange(T+1)
ax.plot(tt, k_rat, 'ko-', lw=2.5, ms=6, label='Rational', zorder=10)
ax.plot(tt, ko, 's-', color=cR(.7), lw=2, ms=6, label='Optimistic GP')
ax.plot(tt, kp_path, 'D-', color=cB(.7), lw=2, ms=6, label='Pessimistic GP')
ax.axhline(k_star, color='gray', ls=':', lw=.8, alpha=.5, label=r'$k^*$')
ax.set(xlabel='Period $t$', ylabel=r'$k_t$'); ax.set_xticks(tt)
ax.set_title('Capital Trajectories'); ax.legend(fontsize=8)

# D – TD errors
ax = fig.add_subplot(gs[1,1])
ax.bar(np.arange(T)-.15, tdo, .3, color=cR(.6), ec=cR(.8), label='Optimistic')
ax.bar(np.arange(T)+.15, tdp, .3, color=cB(.6), ec=cB(.8), label='Pessimistic')
ax.axhline(0, color='k', lw=.8)
ax.set(xlabel='Period $t$', ylabel=r'TD Error $\delta_t$')
ax.set_xticks(np.arange(T)); ax.set_title('Temporal Difference Errors'); ax.legend(fontsize=8)

# E – initial vs final
ax = fig.add_subplot(gs[2,:])
ax.plot(k_grid, kprime_rational, color=blk, lw=3, label=r'Rational $k_{t+1}^*$', zorder=10)
ax.plot(k_grid, k_grid, 'k--', lw=.8, alpha=.4)
ax.plot(k_grid, po[0], color=cR(.4), lw=1.5, ls='--', label=r'Optimistic $t\!=\!0$')
ax.plot(k_grid, po[T], color=cR(.9), lw=2, label=rf'Optimistic $t\!=\!{T}$')
ax.plot(k_grid, pp[0], color=cB(.4), lw=1.5, ls='--', label=r'Pessimistic $t\!=\!0$')
ax.plot(k_grid, pp[T], color=cB(.9), lw=2, label=rf'Pessimistic $t\!=\!{T}$')
ax.axvline(k_star, color='gray', ls=':', lw=.8, alpha=.5)
ax.set(xlabel=r'$k_t$', ylabel=r'$k_{t+1}$', xlim=(k_grid[0],k_grid[-1]), ylim=(k_grid[0],k_grid[-1]))
ax.set_title(rf'Policy Convergence: $t=0$ vs $t={T}$')
ax.legend(fontsize=8, ncol=3, loc='upper left')

plt.savefig('gp_investment_learning.png', dpi=150, bbox_inches='tight')
plt.savefig('gp_investment_learning.pdf', bbox_inches='tight')
print("\nSaved: gp_investment_learning.{png,pdf}")
plt.show()