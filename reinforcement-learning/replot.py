"""
Cleaner replot for the RL reward-distribution figure
(`figures/performance_comparison.png`).

Takes a dict mapping method name to a 1D array of episode rewards (one per
held-out test scenario, as produced by the course evaluation notebook) and
renders an overlaid histogram with mean markers and a shared bin grid.

Usage:

    from replot import replot_rl
    replot_rl({
        "Multi-Restart SA (ours)": ours_rewards,    # length-100 array
        "(s,S) heuristic":         sS_rewards,
        "REINFORCE w/ baseline":   reinforce_rewards,
        "Vanilla SA":              sa_rewards,
    }, save_path="figures/performance_comparison.png")

Run the module directly to render a synthetic-data preview of the styling:

    python replot.py
"""

import numpy as np
import matplotlib.pyplot as plt


_DEFAULT_COLORS = {
    "Multi-Restart SA (ours)": "#1f77b4",
    "(s,S) heuristic":         "#d62728",
    "REINFORCE w/ baseline":   "#2ca02c",
    "Vanilla SA":              "#ff7f0e",
}


def replot_rl(rewards_dict, save_path="performance_comparison.png",
              title=None, n_bins=22):
    """Render the reward-distribution figure from a dict of method -> rewards array."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in rewards_dict.values()])
    bins = np.linspace(float(np.min(all_vals)), float(np.max(all_vals)), n_bins)

    for name, vals in rewards_dict.items():
        vals = np.asarray(vals, dtype=float)
        color = _DEFAULT_COLORS.get(name, None)

        ax.hist(vals, bins=bins, density=True, alpha=0.45,
                edgecolor="black", linewidth=0.6, color=color, label=name)

        mean = float(np.mean(vals))
        ax.axvline(mean, color=color, linestyle=":", linewidth=2)

    ax.set_xlabel("Episode reward (MU)", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(title or
                 "Episode reward distribution across 100 held-out demand scenarios",
                 fontsize=13, pad=12)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(-4, 4))
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = {
        "Multi-Restart SA (ours)": rng.normal(loc=10100, scale=600,  size=100),
        "(s,S) heuristic":         rng.normal(loc=11200, scale=550,  size=100),
        "REINFORCE w/ baseline":   rng.normal(loc=7800,  scale=900,  size=100),
        "Vanilla SA":              rng.normal(loc=8900,  scale=1500, size=100),
    }
    replot_rl(demo, save_path="performance_comparison_demo.png")
