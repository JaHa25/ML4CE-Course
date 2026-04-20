# ML for Chemical Engineering — ICL

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Course](https://img.shields.io/badge/Imperial-ML4CE%202024%2F25-9c1f3c)

> Two ranked-submission coursework projects from Imperial's
> **Machine Learning for Chemical Engineering (ML4CE)** module:
> a **Batch Bayesian Optimiser** for bioreactor titre maximisation,
> and a **Multi-Restart Simulated Annealer** for inventory policy learning.
> Both algorithms were evaluated on held-out environments by the course's
> blind grading platform.

<p align="center">
  <img src="bayesian-optimization/figures/optimization_progress.png" width="48%" alt="BO convergence"/>
  &nbsp;
  <img src="reinforcement-learning/figures/performance_comparison.png" width="48%" alt="RL reward distribution"/>
</p>

*Left: Batch BO converges from a 47.3 g/L initial best to 318.6 g/L within the
60 s grading window. Right: Multi-Restart SA (blue) matches the (s,S) oracle
(red) on held-out demand scenarios, well above the REINFORCE baseline (green).*

## Project 1 — Batch Bayesian Optimization for Bioreactor Titre

**Problem.** Tune five continuous bioreactor inputs (temperature, pH, three feed
rates) and one categorical input (cell type) to maximise product titre.
Tight budget: 6 initial samples + 15 batches of 5 + **60 s** wall-clock per run.

**Approach.**
- Mixed-input Gaussian Process: RBF on continuous dims combined multiplicatively with a categorical kernel on cell type ($\theta_{\text{cat}}=0.25$), so knowledge transfers between cell types without forcing a shared response.
- UCB acquisition with adaptive $\beta$ (3.5 → 1.3) and adaptive length scales; auto-boosted when no improvement for 3 batches or one cell type dominates.
- Greedy batch construction with spatial repulsion ($\exp(-d_{\min}/r)$) plus a 5 % UCB bonus for cell types not yet in the current batch.
- Sobol candidates globally; cross-cell-type local search around the running best (70 % best type, 30 % others).

**Key idea — *cross-cell-type local sampling*.** When refining around the best
point, 30 % of local candidates use the *other* cell types. This catches cases
where a different cell type would actually win at similar conditions, something
a naive local search would miss entirely, and the move that gave us the largest
single jump in performance during development.

![Batch BO convergence](bayesian-optimization/figures/optimization_progress.png)

*Best-so-far titre (red) and per-batch maximum (bars) across 15 batches. The
shaded region marks iterations 3–15, which the course evaluation platform
counts toward the final score. The biggest jump comes at batch 5, after the
local-search component activates at iteration 3.*

**What didn't work.**
- *Single-cell-type local search.* Our first version only refined around the best point's own cell type. It plateaued early, a clear indicator that the cell type itself was a local-optimum trap.
- *Fully joint batch optimisation.* Optimising the batch as a single high-dimensional decision was prohibitively slow under the 60 s budget; sequential greedy with a repulsion penalty closed most of the gap with much less compute.
- *Larger candidate pools late in the run.* Once the GP localised the optimum, scaling the global Sobol pool past ~5 000 candidates added latency without finding new regions worth sampling.

→ [`bayesian-optimization/`](bayesian-optimization/) — code and full report.

---

## Project 2 — Multi-Restart Simulated Annealing for Inventory Management

![Three-echelon supply chain](reinforcement-learning/figures/SCstructure.png)

*Three-echelon supply chain (factory → distribution centre → retailers) under
stochastic Poisson demand. Diagram from the
[ML4CE course materials](https://github.com/OptiMaL-PSE-Lab/Imperial-ML4CE-Course).*

**Problem.** Learn an ordering policy across a three-echelon supply chain with
lead-time delays and stochastic demand. Budget per run: 5000 episodes, 5 min
wall-clock.

**Approach.**
- Direct, derivative-free optimisation of the policy network's ~500 weights, treating it as a black-box parameter vector and avoiding noisy gradient estimates.
- Gaussian perturbation + Metropolis acceptance with cooling temperature $T_k = T_0 / (1 + 0.5 k)$, $T_0 = 5\!\times\!10^5$; step size $\sigma$ shrinks linearly from 0.4 to 0.2 within each restart.
- 10 independent SA restarts; explore probability decays from 0.8 to 0.2 across restarts. Later restarts perturb the running best with $\sigma = 0.05$.
- Mean over 5 episodes per fitness evaluation, which cut single-evaluation reward variance from $\approx$2 000 to $\approx$900.

**Key idea — *restart with explore→exploit scheduling*.** Single SA runs
plateaued around reward 7000. Independent restarts that progressively shift from
random Xavier initialisation to small perturbations of the running best lifted
the mean above 10,000, a bigger gain than any single hyperparameter we tuned.

![RL reward distribution](reinforcement-learning/figures/performance_comparison.png)

*Distribution of per-episode rewards over 100 held-out demand scenarios.
Multi-Restart SA (blue) sits in the same range as the (s,S) heuristic (red),
which is a hand-engineered oracle with known optimal structure; our policy is
fully learned. REINFORCE-with-baseline (green) trains on the same env but
plateaus far below.*

**What didn't work.**
- *REINFORCE first.* Policy-gradient on the same network plateaued around reward 7800. Stochastic demand made the gradient estimator too noisy to reliably climb the reward landscape.
- *Single-restart SA.* Long single runs got stuck in local optima around reward 7000 regardless of cooling schedule; the landscape has multiple basins that temperature alone can't escape.
- *Fewer, longer restarts.* Reducing $N$ from 10 to 3 with a larger per-restart budget consistently underperformed: diversity of starting points mattered more than depth within a single annealing run.

→ [`reinforcement-learning/`](reinforcement-learning/) — code and full report.

---

## Repository layout

```
.
├── bayesian-optimization/
│   ├── batch_bayesian_optimization.py   # GP + UCB + batch selection
│   ├── report.pdf
│   └── figures/optimization_progress.png
└── reinforcement-learning/
    ├── simulated_annealing_policy_opt.py  # multi-restart SA over NN weights
    ├── report.pdf
    └── figures/{SCstructure,performance_comparison}.png
```

## Reproducing the results

The code depends on the course-provided environment, simulator, and policy
network architecture, which are **not redistributed** here.

```bash
# Clone the course repo
git clone https://github.com/OptiMaL-PSE-Lab/Imperial-ML4CE-Course
cd Imperial-ML4CE-Course

# Set up the conda environments (one per project)
conda env create -f BatchBayesianOptimization/ml4ce_bo.yml
conda env create -f ReinforcementLearning/ml4ce_rl.yml
```

Then drop the cleaned algorithm files into the slots the course notebooks expect:

| Course slot | File from this repo |
|---|---|
| `BatchBayesianOptimization/algorithms/your_algorithm.py` | [`bayesian-optimization/batch_bayesian_optimization.py`](bayesian-optimization/batch_bayesian_optimization.py) |
| `ReinforcementLearning/algorithms/your_algorithm.py` | [`reinforcement-learning/simulated_annealing_policy_opt.py`](reinforcement-learning/simulated_annealing_policy_opt.py) |

Run the course notebooks (`MLCE_Coursework2025_BatchBO.ipynb`,
`ML4CE_RL_INV_CW.ipynb`) to reproduce the results.

## Authors

- **Jakob Elias Hammerschmidt**
- **Marc Al Hachem**
- **Seif Ahmed Moheb Elmehelmy**

## Acknowledgments

Imperial College London — OptiMaL-PSE Lab —
Course materials, simulators, and benchmarking infrastructure from
[OptiMaL-PSE-Lab/Imperial-ML4CE-Course](https://github.com/OptiMaL-PSE-Lab/Imperial-ML4CE-Course).

## License

Code: [MIT](LICENSE). Reports and figures: © the authors.
