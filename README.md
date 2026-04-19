# ML for Chemical Engineering — Imperial College London

Two coursework projects from the **ML4CE** course at Imperial College London,
applying ML-based optimization to chemical engineering problems.

Both algorithms were submitted to the course's blind evaluation platform and
ranked competitively against other student teams.

---

## Project 1: Batch Bayesian Optimization for Bioreactor Titre Maximisation

Optimise five continuous bioreactor conditions (temperature, pH, three feed
rates) and one categorical variable (cell type) to maximise product titre.

**Approach**
- Custom Gaussian Process with a mixed kernel: RBF on continuous inputs,
  multiplicative categorical kernel on cell type
- Adaptive length scales (wider early, narrower as data grows)
- UCB acquisition with a decaying exploration coefficient β
- Greedy batch selection with spatial repulsion + cell-type diversity bonus
- Sobol candidates globally + local samples around the running best

**Constraints** — 6 initial samples, 15 batches of 5, 60 s wall-clock budget per run.

**Result** — best titre **318.6 g/L** on the held-out evaluation environment.

![BO convergence](bayesian-optimization/figures/optimization_progress.png)

See [`bayesian-optimization/`](bayesian-optimization/) for the code and full report.

---

## Project 2: Multi-Restart Simulated Annealing for Inventory Management

Learn an ordering policy for a three-echelon supply chain (manufacturer →
distribution centre → retailers) under stochastic Poisson demand.

**Approach**
- Simulated Annealing on the weights of a fixed-architecture policy network
- Gaussian perturbations with a step size that shrinks over each restart
- Metropolis acceptance with a cooling temperature schedule
- 10 independent restarts with explore/exploit scheduling
  (early restarts re-initialise weights, later restarts perturb the best so far)
- Fitness is averaged over 5 episodes per evaluation to control reward variance

**Constraints** — 5000 episodes, 5 min wall-clock budget per run.

**Result** — average reward ≈ **10,100** vs the REINFORCE-with-baseline benchmark at ≈ 7,800.

See [`reinforcement-learning/`](reinforcement-learning/) for the code and full report.

---

## Setup & Dependencies

Both algorithms depend on course-provided environments and utility modules
that are **not redistributed here**. To run the code, set up the course repo
alongside this one:

- Course repo: <https://github.com/OptiMaL-PSE-Lab/Imperial-ML4CE-Course>
- BO algorithm slot: `BatchBayesianOptimization/algorithms/your_algorithm.py`
- RL algorithm slot: `ReinforcementLearning/algorithms/your_algorithm.py`

Drop the cleaned files in the corresponding slots, install the conda
environment provided by the course (`ml4ce_bo.yml` / `ml4ce_rl.yml`), and run
the course notebooks to reproduce the results.

---

## Authors

- Jakob Elias Hammerschmidt
- Marc Al Hachem
- Seif Ahmed Moheb Elmehelmy

## Acknowledgments

Imperial College London — OptiMaL-PSE Lab — ML4CE 2024/25.

## License

[MIT](LICENSE)
