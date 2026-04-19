"""
Batch Bayesian Optimisation for bioreactor titre maximisation.
Mixed-input GP (continuous + categorical cell type) with UCB acquisition
and greedy batch selection under a 60s evaluation budget.
"""

from datetime import datetime

import numpy as np
import scipy.linalg
import sobol_seq


def generate_sobol_candidates(n_candidates, bounds, celltypes):
    """Quasi-random candidates over the 5 continuous dims + cell type."""
    sobol_raw = sobol_seq.i4_sobol_generate(6, n_candidates)

    idx = np.arange(n_candidates)
    np.random.shuffle(idx)
    sobol_raw = sobol_raw[idx, :]

    candidates = []
    for i in range(n_candidates):
        cont = []
        for d in range(5):
            lb, ub = bounds[d]
            cont.append(lb + sobol_raw[i, d] * (ub - lb))

        u_cat = sobol_raw[i, 5]
        cat_idx = min(int(np.floor(u_cat * len(celltypes))), len(celltypes) - 1)
        candidates.append(cont + [celltypes[cat_idx]])

    return candidates


def generate_local_candidates(best_point, bounds, celltypes, n_local):
    """Local samples around the current best, with cell-type variants."""
    best_cont = np.array(best_point[:5], dtype=float)
    best_cat = best_point[5]

    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    ranges = ub - lb

    delta = np.maximum(0.12 * ranges, 0.03 * ranges)
    lows = np.maximum(lb, best_cont - delta)
    highs = np.minimum(ub, best_cont + delta)

    n_main = int(n_local * 0.7)
    u = np.random.rand(n_main, 5)
    samples = lows + u * (highs - lows)
    local_cands = [list(samples[i]) + [best_cat] for i in range(n_main)]

    n_per_other = (n_local - n_main) // max(1, len(celltypes) - 1)
    for ct in celltypes:
        if ct != best_cat:
            u_var = np.random.rand(n_per_other, 5)
            samples_var = lows + u_var * (highs - lows)
            for i in range(n_per_other):
                local_cands.append(list(samples_var[i]) + [ct])

    return local_cands


class MixedGP:
    """GP with RBF kernel on continuous dims, multiplicative categorical kernel on cell type."""

    def __init__(self, X_train, y_train, bounds, base_noise_std=0.08, theta_cat=0.25):
        self.X_train = np.array(X_train, dtype=object)

        y_raw = np.array(y_train).flatten().astype(float)
        self.y_mean = float(np.mean(y_raw))
        self.y_std = max(float(np.std(y_raw)), 1e-8)
        self.y_train = (y_raw - self.y_mean) / self.y_std

        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])

        # Length scales widen early, narrow as data accumulates
        n_train = len(X_train)
        data_frac = min(n_train / 60.0, 1.0)
        base_scale = 0.35 - 0.15 * data_frac

        ranges = self.ub - self.lb
        self.length_scales = np.clip(base_scale * ranges, 0.10, 0.50)

        self.signal_std = 1.0
        self.noise_std = max(base_noise_std, 0.05)
        self.theta_cat = theta_cat

        self.K_train, self.L, self.alpha = self._fit()

    def _normalise_cont(self, X):
        Xc = np.array([row[:5] for row in X], dtype=float)
        return (Xc - self.lb) / (self.ub - self.lb)

    def _kernel(self, X1, X2):
        X1_c = self._normalise_cont(X1)
        X2_c = self._normalise_cont(X2)
        diff = X1_c[:, None, :] - X2_c[None, :, :]
        sqdist = np.sum((diff / self.length_scales) ** 2, axis=2)
        K_cont = np.exp(-0.5 * sqdist)

        cat1 = np.array([str(x[5]) for x in X1])
        cat2 = np.array([str(x[5]) for x in X2])
        same_cat = (cat1[:, None] == cat2[None, :])
        K_cat = np.where(same_cat, 1.0, self.theta_cat)

        return K_cont * K_cat

    def _fit(self):
        n = len(self.X_train)
        K = self._kernel(self.X_train, self.X_train)
        K += (self.noise_std ** 2 + 1e-6) * np.eye(n)

        try:
            L = np.linalg.cholesky(K)
            alpha = scipy.linalg.cho_solve((L, True), self.y_train)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = scipy.linalg.cho_solve((L, True), self.y_train)
            except np.linalg.LinAlgError:
                L, alpha = None, None

        return K, L, alpha

    def predict_ucb(self, X_new, beta):
        X_new = np.array(X_new, dtype=object)

        if self.L is None:
            return np.ones(len(X_new)) * (self.y_mean + 2 * self.y_std * beta)

        K_star = self._kernel(X_new, self.X_train)
        mean = K_star @ self.alpha

        v = scipy.linalg.solve_triangular(self.L, K_star.T, lower=True)
        var = self.signal_std ** 2 - np.sum(v ** 2, axis=0)
        std = np.sqrt(np.maximum(var, 1e-10))

        mean_out = self.y_mean + self.y_std * mean
        std_out = self.y_std * std

        return mean_out + beta * std_out


def select_batch(gp, candidates, batch_size, beta, bounds, celltypes,
                 penalty_radius, penalty_strength):
    """Greedy UCB batch selection with spatial repulsion + cell-type diversity bonus."""
    candidates = np.array(candidates, dtype=object)
    ucb = gp.predict_ucb(candidates, beta)

    cands_cont = np.array([c[:5] for c in candidates], dtype=float)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    cands_norm = (cands_cont - lb) / (ub - lb)

    cand_cats = np.array([str(c[5]) for c in candidates])

    batch = []
    batch_cats = []
    taken = np.zeros(len(candidates), dtype=bool)

    for b_idx in range(batch_size):
        adj = ucb.copy()

        if batch:
            batch_norm = (np.array([b[:5] for b in batch]) - lb) / (ub - lb)
            dists = np.linalg.norm(cands_norm[:, None, :] - batch_norm[None, :, :], axis=2)
            min_dist = np.min(dists, axis=1)
            spatial_penalty = np.exp(-min_dist / penalty_radius)
            adj = adj * (1 - penalty_strength * spatial_penalty)

            missing_cats = set(celltypes) - set(batch_cats)
            if missing_cats and b_idx < batch_size - 1:
                cat_bonus = np.array([0.05 if c in missing_cats else 0.0 for c in cand_cats])
                adj = adj + cat_bonus * abs(np.mean(ucb))

        adj[taken] = -np.inf
        idx = np.argmax(adj)
        batch.append(list(candidates[idx]))
        batch_cats.append(str(candidates[idx][5]))
        taken[idx] = True

    return batch


class BO:
    """Batch BO loop: fit GP, draw global+local candidates, select diverse batch via UCB."""

    def __init__(self, X_initial, X_searchspace, iterations, batch, objective_func):
        start_time = datetime.timestamp(datetime.now())

        self.X_initial = X_initial
        self.X_searchspace = X_searchspace
        self.iterations = iterations
        self.batch = batch

        bounds = [
            (30, 40),   # Temperature (°C)
            (6, 8),     # pH
            (0, 50),    # Feed 1 (mM) at t=40h
            (0, 50),    # Feed 2 (mM) at t=80h
            (0, 50),    # Feed 3 (mM) at t=120h
        ]
        celltypes = ['celltype_1', 'celltype_2', 'celltype_3']

        self.Y = objective_func(self.X_initial)
        self.time = [datetime.timestamp(datetime.now()) - start_time]
        self.time += [0] * (len(self.X_initial) - 1)
        start_time = datetime.timestamp(datetime.now())

        all_X = list(self.X_initial)
        all_Y = list(self.Y)
        best_so_far = float(np.max(all_Y))
        no_improve = 0

        for it in range(iterations):
            # Stop at 52s to stay under the 60s evaluation budget
            elapsed = sum(self.time) + (datetime.timestamp(datetime.now()) - start_time)
            if elapsed > 52:
                break

            noise_estimate = 0.08
            if len(all_Y) > 10:
                y_range = max(all_Y) - min(all_Y)
                noise_estimate = max(0.05, min(0.15, 0.08 * y_range / max(1.0, best_so_far)))

            gp = MixedGP(all_X, all_Y, bounds,
                         base_noise_std=noise_estimate, theta_cat=0.25)

            # beta decays from 3.5 (explore) to 1.3 (exploit)
            frac = it / max(1, iterations - 1)
            beta = max(1.3, 3.5 - 2.2 * frac)
            p_radius = max(0.08, 0.22 - 0.14 * frac)

            if it <= 2:
                n_global = 12000
            elif it <= 5:
                n_global = 8000
            else:
                n_global = 5000

            if it >= 2:
                best_idx = int(np.argmax(all_Y))
                n_local = 4000 if it >= 5 else 2500
                local_cands = generate_local_candidates(
                    all_X[best_idx], bounds, celltypes, n_local
                )
            else:
                local_cands = []

            # Boost exploration if stuck or one cell type dominates
            if no_improve >= 3:
                beta = min(4.0, beta * 1.4)

            if 2 <= it <= 6:
                best_ct = all_X[int(np.argmax(all_Y))][5]
                recent_cts = [str(x[5]) for x in all_X[-10:]]
                ct_dominance = recent_cts.count(str(best_ct)) / len(recent_cts)
                if ct_dominance > 0.85:
                    beta = min(4.0, beta * 1.2)

            global_cands = generate_sobol_candidates(n_global, bounds, celltypes)
            all_cands = global_cands + local_cands

            next_batch = select_batch(
                gp, all_cands, self.batch, beta, bounds, celltypes,
                p_radius, 0.45
            )

            results = objective_func(next_batch)
            all_X.extend(next_batch)
            all_Y = np.concatenate([all_Y, results])
            self.Y = np.concatenate([self.Y, results])

            global_max = float(np.max(all_Y))
            if global_max > best_so_far + 1e-6:
                best_so_far = global_max
                no_improve = 0
            else:
                no_improve += 1

            self.time += [datetime.timestamp(datetime.now()) - start_time]
            self.time += [0] * (len(results) - 1)
            start_time = datetime.timestamp(datetime.now())


if __name__ == "__main__":
    # objective_func is provided by the course environment (see README).
    from objective_func import objective_func  # type: ignore

    bounds = [(30, 40), (6, 8), (0, 50), (0, 50), (0, 50)]
    celltypes = ['celltype_1', 'celltype_2', 'celltype_3']

    sobol_points = sobol_seq.i4_sobol_generate(5, 6)
    X_initial = []
    for i in range(6):
        temp = bounds[0][0] + sobol_points[i, 0] * (bounds[0][1] - bounds[0][0])
        pH = bounds[1][0] + sobol_points[i, 1] * (bounds[1][1] - bounds[1][0])
        f1 = bounds[2][0] + sobol_points[i, 2] * (bounds[2][1] - bounds[2][0])
        f2 = bounds[3][0] + sobol_points[i, 3] * (bounds[3][1] - bounds[3][0])
        f3 = bounds[4][0] + sobol_points[i, 4] * (bounds[4][1] - bounds[4][0])
        celltype = celltypes[i % len(celltypes)]
        X_initial.append([temp, pH, f1, f2, f3, celltype])

    temp = np.linspace(30, 40, 5)
    pH = np.linspace(6, 8, 5)
    f1 = np.linspace(0, 50, 5)
    f2 = np.linspace(0, 50, 5)
    f3 = np.linspace(0, 50, 5)
    X_searchspace = [[a, b, c, d, e, f]
                     for a in temp for b in pH for c in f1 for d in f2 for e in f3
                     for f in celltypes]

    BO_m = BO(X_initial, X_searchspace, 15, 5, objective_func)
