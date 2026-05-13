# eo/eo_simple.py
import random
import numpy as np
from copy import deepcopy

def initialize_population(pop_size, dim, low=-3.0, high=3.0):
    return [np.random.uniform(low, high, size=dim) for _ in range(pop_size)]

def evaluate_population(pop, decode_fn, fitness_fn):
    results = []
    for p in pop:
        aln = decode_fn(p)
        fit, extra = fitness_fn(aln)
        results.append((p, fit, aln, extra))
    return results

def eo_optimize(decode_fn, fitness_fn, dim, pop_size=20, max_iter=100,
                low=-3.0, high=3.0, eq_pool_size=4, verbose=False):
    """
    Simplified EO:
    - population: vectors in R^dim
    - decode_fn(vector) -> alignment (list of strings)
    - fitness_fn(alignment) -> (fitness_scalar, extras)
    Returns best_alignment, best_vector, history
    """
    pop = initialize_population(pop_size, dim, low, high)
    history = []
    # evaluate
    evaluated = evaluate_population(pop, decode_fn, fitness_fn)
    for it in range(max_iter):
        # sort by fitness desc
        evaluated.sort(key=lambda x: x[1], reverse=True)
        eq_pool = [evaluated[i][0] for i in range(min(eq_pool_size, len(evaluated)))]
        new_pop = []
        for idx, (vec, fit, aln, extra) in enumerate(evaluated):
            # pick random equilibrium candidate
            eq = random.choice(eq_pool)
            F = np.random.uniform(0,1, size=dim)
            r = np.random.uniform(0,1, size=dim)
            # update rule (simplified): move toward eq with random factor and decaying influence
            decay = 1.0 - (it / max_iter)
            new_vec = vec + decay * (r * (eq - vec) * F)
            # clamp
            new_vec = np.clip(new_vec, low, high)
            new_pop.append(new_vec)
        # re-evaluate
        evaluated = evaluate_population(new_pop, decode_fn, fitness_fn)
        best = max(evaluated, key=lambda x: x[1])
        history.append(best[1])
        if verbose and (it % max(1, max_iter//10) == 0):
            print(f"EO iter {it}/{max_iter} best fitness {best[1]:.4f}")
    # final best
    evaluated.sort(key=lambda x: x[1], reverse=True)
    best_vec, best_fit, best_aln, best_extra = evaluated[0]
    return {"best_vector": best_vec, "best_fitness": best_fit, "best_alignment": best_aln, "history": history}
