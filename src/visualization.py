import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, make_interp_spline

def plot_phase(k_grid, policies, fixed_points=None, labels=None, colors=None):
    """
    Economist-style k vs k' phase diagram.
    
    policies : list of arrays (each same length as k_grid)
    labels   : list of str
    colors   : list of str
    """
    if not isinstance(policies, list):
        policies = [policies]
    labels = labels or [f"Policy {i+1}" for i in range(len(policies))]
    colors = colors or ["#1a1a2e", "#c0392b", "#2471a3", "#1e8449", "#884ea0"]

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)

    # 45-degree line
    ax.plot(k_grid, k_grid, color="#999999", ls="--", lw=0.8, label=r"$k' = k$", zorder=1)

    # Policy curves
    for i, pol in enumerate(policies):
            ax.plot(k_grid, pol, color=colors[i % len(colors)], lw=2,
                    label=labels[i], zorder=2 + i)
            
    # Fixed points
    if fixed_points is not None:
        for i, fp in enumerate(fixed_points):
            ax.scatter(fp, fp, color=colors[i % len(colors)], s=25, edgecolor="white",
                       label=f"Fixed Point of {labels[i]}", zorder=3 + len(policies) + i)

    # Grid
    ax.grid(True, which="major", color="#d4d4d4", ls="-", lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Kill box
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", which="both", length=0, labelsize=9, colors="#333333")

    ax.axhline(y=ax.get_ylim()[0], color="#333333", lw=0.8)
    ax.axvline(x=ax.get_xlim()[0], color="#333333", lw=0.8)

    ax.set_xlabel(r"$k_t$", fontsize=12, color="#1a1a1a", labelpad=8)
    ax.set_ylabel(r"$k_{t+1}$", fontsize=12, color="#1a1a1a", rotation=0, labelpad=15)
    
    # Legend — flat box, no shadow, upper left
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc", facecolor="white",
              framealpha=1.0, shadow=False, fancybox=False, fontsize=9, labelcolor="#333333")

    fig.tight_layout()
    return fig, ax