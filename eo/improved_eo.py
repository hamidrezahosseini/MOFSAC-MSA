# eo/improved_eo.py 
import numpy as np
import time
import math
from copy import deepcopy
from scipy.stats import levy

class ImprovedEO:
    def __init__(self, decode_fn, fitness_fn, spec):
        self.decode_fn = decode_fn
        self.fitness_fn = fitness_fn
        self.spec = spec
        
    def optimize(self, pop_size=50, max_iter=100, low=-5.0, high=5.0, 
                 seed=None, verbose=True):
        """
        بهبود یافته EO با استراتژی‌های پیشرفته
        """
        rng = np.random.default_rng(seed)
        dim = self.spec['dim']
        
        # 1. مقداردهی اولیه هوشمندانه‌تر
        population = []
        for _ in range(pop_size):
            # استفاده از توزیع نرمال به جای یکنواخت
            vec = rng.normal(0, 1, size=dim)
            vec = np.clip(vec, low, high)
            population.append(vec)
        
        # 2. ارزیابی اولیه
        best_solution = None
        best_fitness = -np.inf
        history = []
        
        for iteration in range(max_iter):
            # محاسبه fitness برای همه
            fitness_values = []
            solutions = []
            
            for vec in population:
                alignment = self.decode_fn(vec, self.spec)
                fitness, info = self.fitness_fn(alignment)
                fitness_values.append(fitness)
                solutions.append((vec, alignment, fitness, info))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = (vec, alignment, fitness, info)
            
            history.append(best_fitness)
            
            # 3. انتخاب بهترین‌ها برای نسل بعدی (الیت‌گرایی)
            sorted_indices = np.argsort(fitness_values)[::-1]
            elite_count = max(2, pop_size // 4)
            elites = [population[i] for i in sorted_indices[:elite_count]]
            
            # 4. تولید نسل جدید
            new_population = elites.copy()
            
            while len(new_population) < pop_size:
                # ترکیب والدین از میان بهترین‌ها
                parent1 = rng.choice(elites)
                parent2 = rng.choice(elites)
                
                # عملگر ترکیب (crossover)
                child = self._crossover(parent1, parent2, rng)
                
                # جهش (mutation)
                if rng.random() < 0.3:
                    child = self._mutate(child, rng, low, high, iteration/max_iter)
                
                # جهش لوی برای تنوع
                if rng.random() < 0.1:
                    child += levy.rvs(size=dim) * 0.01
                
                child = np.clip(child, low, high)
                new_population.append(child)
            
            population = new_population
            
            if verbose and (iteration % 10 == 0 or iteration == max_iter-1):
                print(f"Iter {iteration+1}/{max_iter}, Best Fitness: {best_fitness:.2f}")
        
        return {
            "best_vector": best_solution[0],
            "best_fitness": best_fitness,
            "best_alignment": best_solution[1],
            "history": history
        }
    
    def _crossover(self, p1, p2, rng):
        """عملگر ترکیب"""
        alpha = rng.random()
        return alpha * p1 + (1 - alpha) * p2
    
    def _mutate(self, vec, rng, low, high, progress):
        """جهش تطبیقی"""
        mutation_rate = 0.1 * (1 - progress)  # با گذر زمان کاهش می‌یابد
        mutation_strength = 0.5 * (1 - progress)
        
        mask = rng.random(size=len(vec)) < mutation_rate
        mutations = rng.normal(0, mutation_strength, size=np.sum(mask))
        vec[mask] += mutations
        return np.clip(vec, low, high)