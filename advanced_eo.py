# advanced_eo.py
import numpy as np
import time
from scipy.stats import levy
from copy import deepcopy

class AdvancedEO:
    """الگوریتم EO پیشرفته با استراتژی‌های متعدد"""
    
    def __init__(self, decode_fn, fitness_fn, spec):
        self.decode_fn = decode_fn
        self.fitness_fn = fitness_fn
        self.spec = spec
        
    def optimize(self, pop_size=60, max_iter=100, low=-5.0, high=5.0,
                 strategy='hybrid', verbose=True, seed=None):
        """
        بهینه‌سازی با استراتژی‌های پیشرفته
        
        strategies:
        - 'hybrid': ترکیب EO با جستجوی محلی
        - 'adaptive': EO تطبیقی با پارامترهای پویا
        - 'ensemble': استفاده از چندین استراتژی موازی
        """
        
        if strategy == 'hybrid':
            return self._hybrid_optimization(pop_size, max_iter, low, high, verbose, seed)
        elif strategy == 'adaptive':
            return self._adaptive_optimization(pop_size, max_iter, low, high, verbose, seed)
        elif strategy == 'ensemble':
            return self._ensemble_optimization(pop_size, max_iter, low, high, verbose, seed)
        else:
            return self._hybrid_optimization(pop_size, max_iter, low, high, verbose, seed)
    
    def _hybrid_optimization(self, pop_size, max_iter, low, high, verbose, seed):
        """EO هیبریدی با جستجوی محلی"""
        rng = np.random.default_rng(seed)
        dim = self.spec['dim']
        
        # مقداردهی اولیه جمعیت
        population = []
        for _ in range(pop_size):
            vec = rng.uniform(low, high, size=dim)
            population.append(vec)
        
        best_solution = None
        best_fitness = -np.inf
        history = []
        
        for iteration in range(max_iter):
            # مرحله 1: ارزیابی
            fitness_values = []
            for idx, vec in enumerate(population):
                alignment = self.decode_fn(vec, self.spec)
                fitness, info = self.fitness_fn(alignment)
                fitness_values.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = (vec.copy(), alignment, fitness, info)
            
            history.append(best_fitness)
            
            # مرحله 2: انتخاب بهترین‌ها (الیت‌گرایی)
            sorted_indices = np.argsort(fitness_values)[::-1]
            elite_count = max(3, pop_size // 3)
            elites = [population[i] for i in sorted_indices[:elite_count]]
            
            # مرحله 3: تولید نسل جدید
            new_population = elites.copy()
            
            while len(new_population) < pop_size:
                # استراتژی‌های مختلف برای تولید فرزند
                strategy = rng.choice(['crossover', 'mutation', 'levy', 'local'])
                
                if strategy == 'crossover':
                    # ترکیب دو والد از میان بهترین‌ها
                    p1, p2 = rng.choice(elites, size=2, replace=False)
                    alpha = rng.random()
                    child = alpha * p1 + (1 - alpha) * p2
                    
                elif strategy == 'mutation':
                    # جهش یک والد
                    parent = rng.choice(elites)
                    child = parent.copy()
                    mutation_mask = rng.random(size=dim) < 0.2
                    child[mutation_mask] += rng.normal(0, 0.5, size=np.sum(mutation_mask))
                    
                elif strategy == 'levy':
                    # جهش لوی
                    parent = rng.choice(elites)
                    child = parent.copy()
                    step = levy.rvs(size=dim) * 0.1
                    child += step
                    
                elif strategy == 'local':
                    # جستجوی محلی اطراف بهترین
                    if best_solution:
                        child = best_solution[0].copy()
                        perturbation = rng.normal(0, 0.3, size=dim)
                        child += perturbation
                    else:
                        child = rng.uniform(low, high, size=dim)
                
                # اعمال کران
                child = np.clip(child, low, high)
                new_population.append(child)
            
            population = new_population
            
            # مرحله 4: اعمال جستجوی محلی بر روی بهترین‌ها (هر 10 iteration)
            if iteration % 10 == 0 and best_solution:
                improved = self._local_search(best_solution[0], best_fitness)
                if improved[2] > best_fitness:
                    best_solution = improved
                    best_fitness = improved[2]
            
            if verbose and (iteration % 20 == 0 or iteration == max_iter-1):
                print(f"Iter {iteration+1}/{max_iter}, Best: {best_fitness:.0f}")
        
        return {
            "best_vector": best_solution[0],
            "best_fitness": best_fitness,
            "best_alignment": best_solution[1],
            "history": history
        }
    
    def _local_search(self, vector, current_fitness, steps=5):
        """جستجوی محلی اطراف یک راه‌حل"""
        best_vector = vector.copy()
        best_alignment = None
        best_fitness = current_fitness
        best_info = {}
        
        rng = np.random.default_rng()
        
        for _ in range(steps):
            # ایجاد تغییرات کوچک
            trial_vector = vector + rng.normal(0, 0.1, size=len(vector))
            trial_vector = np.clip(trial_vector, -5, 5)
            
            # ارزیابی
            alignment = self.decode_fn(trial_vector, self.spec)
            fitness, info = self.fitness_fn(alignment)
            
            if fitness > best_fitness:
                best_vector = trial_vector.copy()
                best_alignment = alignment
                best_fitness = fitness
                best_info = info
        
        return (best_vector, best_alignment, best_fitness, best_info)
    
    def _adaptive_optimization(self, pop_size, max_iter, low, high, verbose, seed):
        """EO تطبیقی با پارامترهای پویا"""
        # پیاده‌سازی مشابه اما با پارامترهای پویا
        pass
    
    def _ensemble_optimization(self, pop_size, max_iter, low, high, verbose, seed):
        """EO ensemble با چندین زیرجمعیت"""
        pass