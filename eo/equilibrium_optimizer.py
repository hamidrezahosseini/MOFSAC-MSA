# eo/equilibrium_optimizer.py
"""
Professional Equilibrium Optimizer (EO) implementation.

API expected:
- decode_fn(vector: np.ndarray) -> alignment (list[str])    # convert candidate vector -> alignment
- fitness_fn(alignment: list[str]) -> (fitness: float, details: dict)

This EO implementation:
- Uses an equilibrium pool formed by the best solutions each iteration (top `eq_pool_size`)
  plus an averaged equilibrium candidate.
- Implements concentration updating and stochastic exploration/exploitation terms inspired by
  the original EO paper (Faramarzi et al., 2020) and common open-source variants.
- Handles variable bounds, elitism, and returns history and best alignment/vector.

Returns a dict:
{
  "best_vector": np.ndarray,
  "best_fitness": float,
  "best_alignment": list[str],
  "history": [best_fitness_each_iter],
  "evaluations": total_model_evaluations
}
"""
from copy import deepcopy
import numpy as np
import math
import time
import random

def _initialize_population(pop_size, dim, low, high, rng):
    return [rng.uniform(low, high, size=dim) for _ in range(pop_size)]

def _clamp(vec, low, high):
    return np.minimum(np.maximum(vec, low), high)

def _evaluate_population(pop, decode_fn, fitness_fn):
    results = []
    for v in pop:
        aln = decode_fn(v)
        fit, info = fitness_fn(aln)
        results.append({"vec": v, "fitness": float(fit), "alignment": aln, "info": info})
    return results

def _build_equilibrium_pool(evaluated, eq_pool_size):
    """
    Build equilibrium pool from evaluated list (dicts with keys: vec, fitness).
    Return list of equilibrium candidate vectors (length eq_pool_size + 1 for average).
    """
    # sort descending by fitness
    sorted_eval = sorted(evaluated, key=lambda x: x["fitness"], reverse=True)
    pool = [sorted_eval[i]["vec"].copy() for i in range(min(eq_pool_size, len(sorted_eval)))]
    # add average of top solutions as an extra equilibrium candidate
    if len(pool) > 0:
        avg = np.mean(pool, axis=0)
        pool.append(avg)
    return pool

def equilibrium_optimizer(
    decode_fn,
    fitness_fn,
    dim,
    pop_size=40,
    max_iter=200,
    low=-3.0,
    high=3.0,
    eq_pool_size=4,
    a1=2.0,
    a2=1.0,
    GP_min=0.5,
    GP_max=0.9,
    seed=None,
    verbose=False,
    max_eval=None
):
    """
    Full EO optimizer.

    Parameters
    ----------
    decode_fn : callable
        vector -> alignment (list[str])
    fitness_fn : callable
        alignment -> (fitness: float, details: dict)
    dim : int
        dimensionality of candidate vectors
    pop_size : int
    max_iter : int
    low, high : float or arrays
        bounds for vector entries (scalars or arrays shape (dim,))
    eq_pool_size : int
        number of best solutions to include in equilibrium pool (plus averaged candidate)
    a1, a2 : float
        control parameters for generation rate and exploitation intensity (paper-inspired)
    GP_min, GP_max : floats in (0,1)
        min & max generation probability (stochastic factor)
    seed : int or None
        random seed for reproducibility
    verbose : bool
    max_eval : int or None
        optional cap on number of fitness evaluations (stops early if reached)
    """

    rng = np.random.default_rng(seed)
    rand = random.Random(seed)

    # allow scalar or array bounds
    if np.isscalar(low):
        low_vec = np.full(dim, low, dtype=float)
    else:
        low_vec = np.array(low, dtype=float)
    if np.isscalar(high):
        high_vec = np.full(dim, high, dtype=float)
    else:
        high_vec = np.array(high, dtype=float)

    # initialize population
    population = _initialize_population(pop_size, dim, low_vec, high_vec, rng)
    evaluated = _evaluate_population(population, decode_fn, fitness_fn)
    eval_count = len(evaluated)

    # track global best
    best = max(evaluated, key=lambda x: x["fitness"])
    global_best_vec = best["vec"].copy()
    global_best_fit = best["fitness"]
    global_best_aln = best["alignment"]

    history = [global_best_fit]

    # main loop
    start_time = time.time()
    for t in range(1, max_iter+1):
        # early stop by evaluations
        if max_eval is not None and eval_count >= max_eval:
            if verbose:
                print(f"EO: reached max_eval={max_eval}; stopping early at iter {t}")
            break

        # build equilibrium pool (best eq_pool_size solutions + average)
        eq_pool = _build_equilibrium_pool(evaluated, eq_pool_size)
        if len(eq_pool) == 0:
            eq_pool = [global_best_vec.copy()]

        new_population = []
        # dynamic parameters
        T = t / max_iter  # normalized iteration (0..1)
        # generation probability (GP) as in paper: adaptively sampled between GP_min..GP_max
        GP = GP_min + (GP_max - GP_min) * rng.random()

        # exploitation/exploration control factor decaying with time
        decay = (1.0 - T)  # simple linear decay; can be replaced with other schedules

        for i, indiv in enumerate(evaluated):
            X = indiv["vec"].copy()
            # pick a random equilibrium candidate from pool
            Xeq = eq_pool[rng.integers(0, len(eq_pool))].copy()

            # random vectors
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            lambda_vec = rng.random(dim)  # lambda in [0,1] vector

            # Generation rate random component (G0 in literature)
            G0 = a1 * (1.0 - 2.0 * rng.random(dim))  # in [-a1,a1]

            # compute F and G as exploration/exploitation modulators
            # F tends to zero as t grows (more exploitation later)
            F = a2 * np.exp(-lambda_vec * T)  # exploitation factor (vector)
            # stochastic perturbation
            rand_pert = (rng.random(dim) - 0.5) * 2.0  # in [-1,1]

            # Concentration updating equation (paper-inspired form):
            # X_new = Xeq + (X - Xeq) * F + G0 * rand_pert * decay * (high-low)
            # where G0 is generation amplitude and decay reduces randomness over time
            range_vec = (high_vec - low_vec)
            X_new = Xeq + (X - Xeq) * F + G0 * rand_pert * decay * range_vec

            # Jumping mechanism with probability GP (to escape local optima)
            jump_mask = rng.random(dim) < GP
            if np.any(jump_mask):
                # random re-initialization of selected dimensions within bounds
                random_component = rng.uniform(low_vec, high_vec)
                X_new[jump_mask] = random_component[jump_mask]

            # boundary handling - clamp
            X_new = _clamp(X_new, low_vec, high_vec)

            new_population.append(X_new)

        # evaluate new population
        evaluated = _evaluate_population(new_population, decode_fn, fitness_fn)
        eval_count += len(evaluated)

        # elitism: keep best between old best and new bests
        combined = evaluated + [{"vec": global_best_vec, "fitness": global_best_fit, "alignment": global_best_aln}]
        best = max(combined, key=lambda x: x["fitness"])
        global_best_vec = best["vec"].copy()
        global_best_fit = best["fitness"]
        global_best_aln = best["alignment"]

        history.append(global_best_fit)

        if verbose and (t % max(1, max_iter//10) == 0 or t==1):
            elapsed = time.time() - start_time
            print(f"EO iter {t}/{max_iter} | best fitness: {global_best_fit:.6f} | evals: {eval_count} | time: {elapsed:.1f}s")

    result = {
        "best_vector": global_best_vec,
        "best_fitness": float(global_best_fit),
        "best_alignment": global_best_aln,
        "history": history,
        "evaluations": eval_count
    }
    return result


# quick self-test (if run directly)
if __name__ == "__main__":
    import numpy as np
    def decode_stub(v):
        # trivial decode: convert vector to two fake sequences of same length (for testing)
        L = max(1, len(v))
        s1 = ''.join(['A' if x>0 else '-' for x in v])
        s2 = ''.join(['A' if x>-0.5 else '-' for x in v])
        return [s1, s2]

    def fitness_stub(aln):
        # simple fitness: maximize number of matches (A/A)
        n = len(aln); L = len(aln[0])
        score = 0
        for c in range(L):
            if all(row[c]==aln[0][c] for row in aln):
                score += 1
        return float(score), {}

    res = equilibrium_optimizer(decode_stub, fitness_stub, dim=10, pop_size=12, max_iter=20, low=-1, high=1, verbose=True, seed=42)
    print("BEST FITNESS:", res["best_fitness"])
