# eo/equilibrium_optimizer_pro.py
import numpy as np
import time
import math
from copy import deepcopy
import random

from scipy.stats import levy

# expects decode_fn and fitness_fn signature as before

def _levy_flight(dim, scale=0.01, rng=np.random):
    # generate levy-stable steps
    # using scipy.stats.levy rvs (heavy-tailed)
    return levy.rvs(size=dim) * scale

def _gaussian_mutation(vec, sigma, rng):
    return vec + rng.normal(0, sigma, size=vec.shape)

def _clamp(vec, lo, hi):
    return np.minimum(np.maximum(vec, lo), hi)

def equilibrium_optimizer_pro(
    decode_fn, fitness_fn, spec,
    pop_size=50, max_iter=250, low=-5.0, high=5.0,
    eq_pool_size=5, p_mut=0.15, p_reseed=0.02, levy_scale=0.02,
    seed=None, verbose=False, max_eval=None
):
    """
    spec: dict returned by build_hybrid_spec (contains dim and offsets etc)
    decode_fn: should accept vector and spec (we will wrap)
    """
    rng = np.random.default_rng(seed)
    rand = random.Random(seed)
    dim = spec['dim']
    low_vec = np.full(dim, low); high_vec = np.full(dim, high)

    def decode_wrapper(v):
        return decode_fn(v, spec, max_shift=20)

    # init pop
    pop = [rng.uniform(low, high, size=dim) for _ in range(pop_size)]
    evaluated = []
    for v in pop:
        aln = decode_wrapper(v)
        f, info = fitness_fn(aln)
        evaluated.append({"vec": v.copy(), "fitness": float(f), "alignment": aln, "info": info})
    eval_count = len(evaluated)
    best = max(evaluated, key=lambda x: x['fitness'])
    global_best = deepcopy(best)
    history = [global_best['fitness']]
    stagnation = 0
    start = time.time()

    for t in range(1, max_iter+1):
        if max_eval is not None and eval_count >= max_eval:
            break

        # build eq pool
        evaluated.sort(key=lambda x: x['fitness'], reverse=True)
        pool = [evaluated[i]['vec'] for i in range(min(eq_pool_size, len(evaluated)))]
        if len(pool) > 0:
            pool.append(np.mean(pool, axis=0))

        new_pop = []
        # for each individual
        for i, ind in enumerate(evaluated):
            X = ind['vec'].copy()
            # pick eq
            Xeq = pool[rng.integers(0, len(pool))].copy() if len(pool)>0 else X.copy()
            dimv = dim

            # exploitation / exploration factors
            r1 = rng.random(dimv)
            r2 = rng.random(dimv)
            lambda_vec = rng.random(dimv)
            T = t / max_iter
            # control coefficients (inspired by EO)
            F = np.exp(-lambda_vec * T)  # decaying factor
            G0 = 2.0 * (rng.random(dimv) - 0.5)  # in [-1,1]*2

            # concentration update (with stronger exploration)
            X_new = Xeq + (X - Xeq) * F + G0 * (rng.random(dimv) - 0.5) * (high_vec - low_vec) * (1.0 - T)

            # Lévy mutation with small prob
            if rng.random() < 0.25:
                X_new += _levy_flight(dimv, scale=levy_scale, rng=rng)

            # gaussian mutation
            if rng.random() < p_mut:
                X_new = _gaussian_mutation(X_new, sigma=0.5*(1.0 - T), rng=rng)

            # occasional large reseed for escape
            if rng.random() < p_reseed:
                mask = rng.random(dimv) < 0.1
                X_new[mask] = rng.uniform(low, high, size=mask.sum())

            # clamp
            X_new = _clamp(X_new, low_vec, high_vec)
            new_pop.append(X_new)

        # evaluate new pop
        evaluated = []
        for v in new_pop:
            aln = decode_wrapper(v)
            f, info = fitness_fn(aln)
            evaluated.append({"vec": v.copy(), "fitness": float(f), "alignment": aln, "info": info})
        eval_count += len(evaluated)

        # elitism
        cand_best = max(evaluated, key=lambda x: x['fitness'])
        if cand_best['fitness'] > global_best['fitness']:
            global_best = deepcopy(cand_best)
            stagnation = 0
        else:
            stagnation += 1

        history.append(global_best['fitness'])

        # adaptive intensification: if stagnation grows, increase mutation
        if stagnation > 12:
            p_mut = min(0.6, p_mut * 1.5)
            # heavy reseed
            if rng.random() < 0.3:
                # replace 10% worst with random
                evaluated.sort(key=lambda x: x['fitness'])
                replace_k = max(1, int(0.1 * len(evaluated)))
                for i in range(replace_k):
                    newv = rng.uniform(low, high, size=dim)
                    aln = decode_wrapper(newv)
                    f, info = fitness_fn(aln)
                    evaluated[i] = {"vec": newv.copy(), "fitness": float(f), "alignment": aln, "info": info}
                # recompute best
                global_best = max(evaluated + [global_best], key=lambda x: x['fitness'])

        if verbose and (t % max(1, max_iter//10) == 0 or t==1):
            elapsed = time.time() - start
            print(f"EO-Pro iter {t}/{max_iter} | best fitness: {global_best['fitness']:.6f} | evals: {eval_count} | time: {elapsed:.1f}s")

    result = {
        "best_vector": global_best['vec'],
        "best_fitness": float(global_best['fitness']),
        "best_alignment": global_best['alignment'],
        "history": history,
        "evaluations": eval_count
    }
    return result
