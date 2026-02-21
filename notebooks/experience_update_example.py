import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

R = 1.05
beta = 1 / R
s1 = 10.0
a = 0.05
sigEps = 1.5
sig0 = 2.0
psiS = 0.1
psiC = 0.1

cstar = R * s1 / (1 + R)

# === RBF Kernel ===
def k(s, c, sp, cp):
    return sig0**2 * np.exp(-psiS * (s - sp)**2 - psiC * (c - cp)**2)

# === TD Error ===
def delta(eps):
    return beta * (
        eps * (1 - a * cstar)
        - a / 2 * eps**2
        + a / 2 * sigEps**2
    )

# === Kalman Gain ===
def alpha_E(eps, s, c):
    num = (
        k(s1, cstar, s, c) -
        beta * k(cstar + eps, cstar + eps, s, c)
    )
    den = (
        k(s1, cstar, s1, cstar) -
        2 * beta * k(s1, cstar, cstar + eps, cstar + eps) +
        beta**2 * k(cstar + eps, cstar + eps, cstar + eps, cstar + eps)
    )
    return num / den


def update(eps, s, c):
    return alpha_E(eps, s, c) * delta(eps)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})


# =============================================
# Plot 1: Update at (s1, cstar) vs epsilon
# =============================================
eps_range = np.linspace(-5, 5, 1000)
updates = [update(e, s1, cstar) for e in eps_range]

fig, ax = plt.subplots(figsize=(8, 4))

# Light gray horizontal gridlines
ax.yaxis.set_major_locator(plt.MaxNLocator(6))
ax.set_axisbelow(True)
ax.grid(axis='y', color='#d4d4d4', linewidth=0.5)

# Remove box
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#333333')
ax.tick_params(left=False, bottom=True,
               colors='#333333')

# Zero lines
ax.axhline(0, color='#999999', linewidth=0.7)
ax.axvline(0, color='#999999', linewidth=0.7,
           linestyle='-')

# Main line
ax.plot(eps_range, updates, color='#e3120b',
        linewidth=2.2)

# Labels
ax.set_xlabel(r'$\varepsilon$', fontsize=13,
              color='#333333')
ax.set_ylabel(
    r'$\hat{Q}^E(s_1, c_1^*) - \hat{Q}_0(s_1, c_1^*)$',
    fontsize=11, color='#333333'
)
# ax.set_title(
#     'Posterior distortion at $(s_1, c_1^*)$',
#     fontsize=14, fontweight='bold', color='#333333',
#     loc='left'
# )

plt.tight_layout()
plt.savefig('update_vs_eps.pgf', dpi=500,
            facecolor='white', bbox_inches='tight')
plt.show()











# =============================================
# Plot 2: Weighted by Gaussian density
# =============================================
phi = (
    1 / (sigEps * np.sqrt(2 * np.pi)) *
    np.exp(-eps_range**2 / (2 * sigEps**2))
)
weighted = np.array(updates) * phi

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(eps_range, weighted, alpha=0.3, color='red')
ax.plot(eps_range, weighted, 'r-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$\varepsilon$', fontsize=14)
ax.set_ylabel(
    r'$\Delta Q \cdot \phi(\varepsilon)$',
    fontsize=12
)
ax.set_title(
    'Integrand of $\mathbb{E}[\Delta Q]$',
    fontsize=13
)
plt.tight_layout()
plt.savefig('weighted_update.png', dpi=150)
plt.show()

# =============================================
# Plot 3: Update surface over (s, c) for
#          three epsilon values
# =============================================
from matplotlib.colors import LinearSegmentedColormap

s_grid = np.linspace(0, 20, 100)
c_grid = np.linspace(0, 20, 100)
SS, CC = np.meshgrid(s_grid, c_grid)

eps_vals = [-2 * sigEps, 0, 2 * sigEps]
labels = [
    r'$\varepsilon = -2\sigma_\varepsilon$',
    r'$\varepsilon = 0$',
    r'$\varepsilon = +2\sigma_\varepsilon$'
]

colors_rwg = ['#c0392b', '#e8e8e8', '#27ae60']
cmap_rwg = LinearSegmentedColormap.from_list(
    'RedGreen', colors_rwg, N=1000
)

def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))**(1/2)

Zs = []
for ev in eps_vals:
    Z_raw = np.array([
        [update(ev, s, c) for s in s_grid]
        for c in c_grid
    ])
    Zs.append(signed_log(Z_raw))

global_vmax = max(
    max(abs(Z.min()), abs(Z.max())) for Z in Zs
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, ev, lab, Z in zip(axes, eps_vals, labels, Zs):
    im = ax.contourf(
        SS, CC, Z, levels=50,
        cmap=cmap_rwg,
        vmin=-global_vmax, vmax=global_vmax
    )

    ax.plot(s1, cstar, 'o', color='#85c1e9',
            markersize=12, markeredgecolor='white',
            markeredgewidth=1.5, zorder=5,
            label=r'$(s_1, c_1^*)$')

    s2 = cstar + ev
    ax.plot(s2, s2, 'o', color='#1a5276',
            markersize=12, markeredgecolor='white',
            markeredgewidth=1.5, zorder=5,
            label=r'$(s_2, s_2)$')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')

    ax.xaxis.set_major_locator(
        plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(
        plt.MaxNLocator(integer=True))
    ax.tick_params(colors='#333333')

    ax.set_xlabel('$s$', fontsize=13,
                  color='#333333')
    ax.set_ylabel('$c$', fontsize=13,
                  color='#333333')
    ax.set_title(lab, fontsize=13,
                 fontweight='bold', color='#333333',
                 loc='left')
    ax.legend(loc='upper left', fontsize=9,
              framealpha=0.8)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label(r'$\mathrm{sgn}(\Delta Q) \cdot \sqrt{\ln(1 + |\Delta Q|)}$',
             fontsize=11, color='#333333')
    cb.ax.tick_params(colors='#333333')

plt.tight_layout()
plt.savefig('update_surface.pgf',
            bbox_inches='tight')
plt.savefig('update_surface.png', dpi=300,
            facecolor='white', bbox_inches='tight')
plt.show()


# =============================================
# Compute E[update] and E[update^2]
# =============================================
def integrand_mean(e):
    return (
        update(e, s1, cstar) *
        1 / (sigEps * np.sqrt(2 * np.pi)) *
        np.exp(-e**2 / (2 * sigEps**2))
    )

def integrand_mse(e):
    return (
        update(e, s1, cstar)**2 *
        1 / (sigEps * np.sqrt(2 * np.pi)) *
        np.exp(-e**2 / (2 * sigEps**2))
    )

E_update, _ = integrate.quad(integrand_mean, -10, 10)
E_update2, _ = integrate.quad(integrand_mse, -10, 10)

print(f"\nc* = {cstar:.4f}")
print(f"E[update] at (s1, c*) = {E_update:.6f}")
print(f"E[update^2] at (s1, c*) = {E_update2:.6f}")
print(f"Std of update = {np.sqrt(E_update2 - E_update**2):.6f}")


from scipy.optimize import minimize_scalar

# Max positive distortion
res_max = minimize_scalar(
    lambda e: -update(e, s1, cstar),
    bounds=(-10, 10), method='bounded'
)

# Max negative distortion
res_min = minimize_scalar(
    lambda e: update(e, s1, cstar),
    bounds=(-10, 10), method='bounded'
)

print(f"Worst positive: eps = {res_max.x:.4f}, "
      f"update = {-res_max.fun:.6f}")
print(f"Worst negative: eps = {res_min.x:.4f}, "
      f"update = {res_min.fun:.6f}")

# Worst absolute
if abs(res_min.fun) > abs(res_max.fun):
    print(f"\nWorst overall: eps = {res_min.x:.4f}")
else:
    print(f"\nWorst overall: eps = {res_max.x:.4f}")




import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# === Parameters ===
beta = 0.95
R = 1.05
s1 = 10.0
sig0 = 2.0
psiS = 0.1
psiC = 0.1
psi = psiS + psiC

# === Optimal policy ===
def cstar(s):
    return (1 - beta) * s

# === True Q* and V* ===
xi = (np.log(1 - beta) / (1 - beta) +
      beta * np.log(R * beta) / (1 - beta)**2)

def Qstar(s, c):
    return (np.log(c) +
            beta / (1 - beta) * np.log(R * (s - c)) +
            xi)

def Vstar(s):
    return 1 / (1 - beta) * np.log(s) + xi

# === Log-Laplace kernel (min-ratio form) ===
def minrat(x, y):
    return np.minimum(x / y, y / x)

def kernel(s1v, c1v, s2v, c2v):
    return (sig0**2 *
            minrat(s1v, s2v)**psiS *
            minrat(c1v, c2v)**psiC)

# === Single-period GP update ===
def gp_update_single(s_dec, c_dec, s_out, c_out,
                     delta_val, s_query, c_query):
    """Kalman gain * delta at query point."""
    k1 = kernel(s_dec, c_dec, s_query, c_query)
    k2 = kernel(s_out, c_out, s_query, c_query)
    k11 = sig0**2  # self-kernel
    k12 = kernel(s_dec, c_dec, s_out, c_out)
    k22 = sig0**2
    num = k1 - beta * k2
    den = k11 - 2 * beta * k12 + beta**2 * k22
    return num / den * delta_val

# === Multi-observation GP update ===
def gp_update_multi(obs_list, s_query, c_query):
    """
    obs_list: list of dicts with keys
      s_dec, c_dec, s_out, c_out, delta
    Returns posterior correction at (s_query, c_query).
    """
    n = len(obs_list)
    if n == 0:
        return 0.0

    # Build K matrix (n x n)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            oi = obs_list[i]
            oj = obs_list[j]
            # Cov(delta_i, delta_j)
            # delta = u(c) + beta Q(s', c') - Q(s, c)
            # Cov = k(s_i,c_i ; s_j,c_j)
            #      - beta k(s_i,c_i ; s'_j,c'_j)
            #      - beta k(s'_i,c'_i ; s_j,c_j)
            #      + beta^2 k(s'_i,c'_i ; s'_j,c'_j)
            K[i, j] = (
                kernel(oi['s_dec'], oi['c_dec'],
                       oj['s_dec'], oj['c_dec']) -
                beta * kernel(oi['s_dec'], oi['c_dec'],
                              oj['s_out'], oj['c_out']) -
                beta * kernel(oi['s_out'], oi['c_out'],
                              oj['s_dec'], oj['c_dec']) +
                beta**2 * kernel(oi['s_out'], oi['c_out'],
                                 oj['s_out'], oj['c_out'])
            )

    # Build k_star vector (n,)
    k_star = np.zeros(n)
    for i in range(n):
        oi = obs_list[i]
        k_star[i] = (
            kernel(oi['s_dec'], oi['c_dec'],
                   s_query, c_query) -
            beta * kernel(oi['s_out'], oi['c_out'],
                          s_query, c_query)
        )

    # Delta vector
    deltas = np.array([o['delta'] for o in obs_list])

    # Solve K w = deltas, return k_star @ w
    try:
        w = np.linalg.solve(K + 1e-6 * np.eye(n), deltas)
        return k_star @ w
    except np.linalg.LinAlgError:
        return 0.0

# === Find optimal c given Q_hat ===
def find_best_c(s_val, obs_list):
    """Maximize Q*(s,c) + GP_correction(s,c) over c."""
    def neg_Qhat(c):
        if c <= 0.01 or c >= s_val - 0.01:
            return 1e10
        corr = gp_update_multi(obs_list, s_val, c)
        return -(Qstar(s_val, c) + corr)

    res = minimize_scalar(neg_Qhat,
                          bounds=(0.01, s_val - 0.01),
                          method='bounded')
    return res.x

# === Simulation ===
def simulate(eps1, n_periods=100):
    """
    Period 1: shock eps1.
    Period 2 onward: eps = 0.
    """
    # Storage
    s_path = np.zeros(n_periods + 1)
    c_path = np.zeros(n_periods)
    delta_path = np.zeros(n_periods)
    c_opt_path = np.zeros(n_periods)
    s_opt_path = np.zeros(n_periods + 1)
    obs_list = []

    s_path[0] = s1
    s_opt_path[0] = s1

    for t in range(n_periods):
        s_t = s_path[t]
        eps_t = eps1 if t == 0 else 0.0

        # Agent's choice
        if t == 0:
            c_t = cstar(s_t)  # correct prior
        else:
            c_t = find_best_c(s_t, obs_list)

        c_path[t] = c_t
        c_opt_path[t] = cstar(s_t)

        # Transition
        s_next = R * (s_t - c_t) * np.exp(eps_t)
        s_path[t + 1] = s_next

        # Counterfactual optimal path
        s_opt_path[t + 1] = (R * (s_opt_path[t] -
                              cstar(s_opt_path[t])) *
                              np.exp(eps_t))

        # Greedy next action under current beliefs
        if t < n_periods - 1:
            c_next = find_best_c(s_next, obs_list)
        else:
            c_next = cstar(s_next)

        # TD error
        Q_current = (Qstar(s_t, c_t) +
                     gp_update_multi(obs_list, s_t, c_t))
        Q_next = (Qstar(s_next, c_next) +
                  gp_update_multi(obs_list, s_next,
                                  c_next))
        delta_t = np.log(c_t) + beta * Q_next - Q_current
        delta_path[t] = delta_t

        # Store observation
        obs_list.append({
            's_dec': s_t,
            'c_dec': c_t,
            's_out': s_next,
            'c_out': c_next,
            'delta': delta_t
        })

    return {
        's': s_path,
        'c': c_path,
        'delta': delta_path,
        'c_opt': c_opt_path,
        's_opt': s_opt_path,
        'obs': obs_list
    }

# === Run for different eps1 ===
eps_values = [-1.0, -0.5, 0.5]
n_per = 40
results = {}
for ev in eps_values:
    print(f"Simulating eps1 = {ev}...")
    results[ev] = simulate(ev, n_per)

# === Plot 1: Belief error at decision point ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Q-hat error at visited states
ax = axes[0]
for ev in eps_values:
    r = results[ev]
    errors = []
    for t in range(n_per):
        corr = gp_update_multi(
            r['obs'][:t+1],
            r['s'][t], r['c'][t])
        errors.append(abs(corr))
    ax.plot(range(n_per), errors,
            label=f'$\\varepsilon_1={ev}$')
ax.set_xlabel('Period $t$')
ax.set_ylabel('$|\\hat{Q}_t - Q^*|$ at visited state')
ax.set_title('Belief error at visited states',
             loc='left', fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 2: Policy distortion
ax = axes[1]
for ev in eps_values:
    r = results[ev]
    distortion = ((r['c'] - r['c_opt']) /
                  r['s'][:n_per])
    ax.plot(range(n_per), distortion,
            label=f'$\\varepsilon_1={ev}$')
ax.axhline(0, color='#999999', linewidth=0.7)
ax.set_xlabel('Period $t$')
ax.set_ylabel('$(\\tilde{c}_t - c_t^*) / s_t$')
ax.set_title('Policy distortion',
             loc='left', fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel 3: Wealth divergence
ax = axes[2]
for ev in eps_values:
    r = results[ev]
    ratio = r['s'][:n_per] / r['s_opt'][:n_per]
    ax.plot(range(n_per), ratio,
            label=f'$\\varepsilon_1={ev}$')
ax.axhline(1, color='#999999', linewidth=0.7)
ax.set_xlabel('Period $t$')
ax.set_ylabel('$s_t / s_t^*$')
ax.set_title('Wealth divergence from optimal',
             loc='left', fontweight='bold')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('learning_trap_simulation.png',
            dpi=300, facecolor='white',
            bbox_inches='tight')
plt.savefig('learning_trap_simulation.pgf',
            bbox_inches='tight')
plt.show()

# === Print summary ===
for ev in eps_values:
    r = results[ev]
    final_corr = gp_update_multi(
        r['obs'], r['s'][-2], r['c'][-1])
    final_dist = ((r['c'][-1] - r['c_opt'][-1]) /
                  r['s'][-2])
    final_wealth = r['s'][-1] / r['s_opt'][-1]
    print(f"\neps1 = {ev:+.1f}:")
    print(f"  Final belief error: {final_corr:.6f}")
    print(f"  Final policy dist:  {final_dist:.6f}")
    print(f"  Final wealth ratio: {final_wealth:.4f}")