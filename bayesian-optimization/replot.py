"""
Cleaner replot for the BO convergence figure (`figures/optimization_progress.png`).

Reads the per-evaluation titre array `Y` (as produced by the course notebook,
e.g. `BO_m.Y`) and renders a higher-contrast convergence chart: faint per-batch
bars in the background, a bold best-so-far line in front, an annotated final
value, and a clearer initial-best reference.

Usage:

    from replot import replot_bo
    replot_bo(BO_m.Y, save_path="figures/optimization_progress.png")

Run the module directly to render a synthetic-data preview of the styling:

    python replot.py
"""

import numpy as np
import matplotlib.pyplot as plt


def replot_bo(Y, batch_size=5, n_init=6,
              save_path="optimization_progress.png", title=None):
    """Render the BO convergence figure from the flat titre array `Y`."""
    Y = np.asarray(Y, dtype=float).ravel()

    init_best = float(np.max(Y[:n_init]))

    n_iters = (len(Y) - n_init) // batch_size
    iterations = np.arange(1, n_iters + 1)

    batch_maxes = np.array([
        np.max(Y[n_init + (i - 1) * batch_size: n_init + i * batch_size])
        for i in iterations
    ])
    best_so_far = np.maximum.accumulate(np.maximum(batch_maxes, init_best))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(iterations, batch_maxes, alpha=0.25, color="steelblue",
           edgecolor="none", label="Per-batch max")

    ax.plot(iterations, best_so_far, "o-", color="#c62828", linewidth=2.5,
            markersize=9, markerfacecolor="white", markeredgewidth=2,
            label="Best so far")

    ax.axhline(y=init_best, color="#555", linestyle="--", linewidth=1.5,
               label=f"Initial best ({init_best:.1f} g/L)")

    ax.axvspan(2.5, n_iters + 0.5, alpha=0.07, color="green",
               label=f"Graded region (iter 3-{n_iters})")

    final = float(best_so_far[-1])
    ax.annotate(f"Final: {final:.1f} g/L",
                xy=(n_iters, final),
                xytext=(max(1, n_iters - 4.5), final * 0.82),
                fontsize=11, color="#c62828",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2))

    ax.set_xlabel("Batch iteration", fontsize=12)
    ax.set_ylabel("Titre (g/L)", fontsize=12)
    ax.set_title(title or
                 "Bioreactor titre vs. evaluation batch (60 s budget, 5 evals/batch)",
                 fontsize=13, pad=12)
    ax.set_xticks(range(1, n_iters + 1))
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Y_demo = np.concatenate([
        rng.uniform(20, 130, size=6),
        rng.uniform(40, 220, size=15 * 5),
    ])
    Y_demo[6 + 4 * 5: 6 + 5 * 5] = rng.uniform(280, 320, size=5)
    Y_demo[6 + 5 * 5:] = np.maximum(Y_demo[6 + 5 * 5:], 318.6)

    replot_bo(Y_demo, save_path="optimization_progress_demo.png")
