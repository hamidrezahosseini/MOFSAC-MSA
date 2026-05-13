# optimized_eo.py
import numpy as np


class OptimizedEO:
    """
    Equilibrium Optimizer برای بهینه‌سازی بردار gap displacement
    """

    def __init__(
        self,
        decode_fn,
        fitness_fn,
        dim,
        pop_size=40,
        max_iter=50,
        use_equilibrium_pool=True,
        seed=None,
    ):
        self.decode_fn = decode_fn
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.use_equilibrium_pool = use_equilibrium_pool
        self.rng = np.random.default_rng(seed)

    def optimize(self, low=-1.0, high=1.0):
        # --- Initialization ---
        population = self.rng.uniform(low, high, (self.pop_size, self.dim))
        fitness = np.zeros(self.pop_size)
        alignments = [None] * self.pop_size

        for i in range(self.pop_size):
            aln = self.decode_fn(population[i])
            f, _ = self.fitness_fn(aln)
            fitness[i] = f
            alignments[i] = aln

        # --- Equilibrium Pool ---
        if self.use_equilibrium_pool:
            eq_indices = np.argsort(fitness)[-4:]
            eq_pool = population[eq_indices]
        else:
            best_idx = np.argmax(fitness)
            eq_pool = population[[best_idx]]

        # --- Main EO Loop ---
        for t in range(self.max_iter):
            F = 2 * (1 - (t / self.max_iter))  # adaptive factor

            for i in range(self.pop_size):
                # انتخاب equilibrium target
                eq = eq_pool[self.rng.integers(len(eq_pool))]

                lambda_vec = self.rng.random(self.dim)
                r = self.rng.random(self.dim)

                new_pos = (
                    eq
                    + F * r * (lambda_vec * (eq - population[i]))
                )

                new_pos = np.clip(new_pos, low, high)

                aln = self.decode_fn(new_pos)
                f, _ = self.fitness_fn(aln)

                if f > fitness[i]:
                    population[i] = new_pos
                    fitness[i] = f
                    alignments[i] = aln

            # --- Update equilibrium pool ---
            if self.use_equilibrium_pool:
                eq_indices = np.argsort(fitness)[-4:]
                eq_pool = population[eq_indices]
            else:
                best_idx = np.argmax(fitness)
                eq_pool = population[[best_idx]]

        best_idx = np.argmax(fitness)
        return {
            "best_alignment": alignments[best_idx],
            "best_fitness": fitness[best_idx],
        }


class RandomOptimizer:
    """
    جستجوی تصادفی ساده (Baseline Ablation)
    """

    def __init__(
        self,
        decode_fn,
        fitness_fn,
        dim,
        pop_size=40,
        max_iter=50,
        seed=None,
    ):
        self.decode_fn = decode_fn
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

    def optimize(self, low=-1.0, high=1.0):
        best_fitness = -np.inf
        best_alignment = None

        total_samples = self.pop_size * self.max_iter

        for _ in range(total_samples):
            vec = self.rng.uniform(low, high, self.dim)
            aln = self.decode_fn(vec)
            f, _ = self.fitness_fn(aln)

            if f > best_fitness:
                best_fitness = f
                best_alignment = aln

        return {
            "best_alignment": best_alignment,
            "best_fitness": best_fitness,
        }
