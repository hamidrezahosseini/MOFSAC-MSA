# rl/simple_environment.py 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

class SimpleEOEnvironment(gym.Env):
    """محیط ساده‌تر برای تنظیم پارامترهای EO"""
    
    def __init__(self, sequences, seed_alignment, spec):
        super().__init__()
        
        self.sequences = sequences
        self.seed_alignment = seed_alignment
        self.spec = spec
        self.seed_sp = sp_score(seed_alignment)
        
        # فقط 4 پارامتر اصلی را تنظیم می‌کنیم
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1, 0.0, 0.0]),  # [pop_size_ratio, iter_ratio, low, high]
            high=np.array([2.0, 2.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        
    def reset(self):
        state = np.zeros(5, dtype=np.float32)
        return state
    
    def step(self, action):
        # تبدیل action به پارامترهای EO
        pop_size = int(20 + 80 * action[0])  # 20-100
        max_iter = int(20 + 80 * action[1])  # 20-100
        low = -10.0 * action[2]  # 0 تا -10
        high = 10.0 * action[3]  # 0 تا 10
        
        # اجرای EO با پارامترهای جدید
        result = self.run_eo(pop_size, max_iter, low, high)
        best_sp = result["best_fitness"]
        
        # محاسبه reward
        improvement = best_sp - self.seed_sp
        reward = improvement / self.seed_sp * 1000  # نرمال‌سازی
        
        # state جدید
        next_state = np.array([
            pop_size / 100,  # نرمال‌شده
            max_iter / 100,
            action[2],
            action[3],
            improvement / self.seed_sp
        ], dtype=np.float32)
        
        done = True  # فقط یک مرحله
        info = {
            "pop_size": pop_size,
            "max_iter": max_iter,
            "low": low,
            "high": high,
            "best_sp": best_sp,
            "improvement": improvement
        }
        
        return next_state, reward, done, info
    
    def run_eo(self, pop_size, max_iter, low, high):
        from eo.improved_eo import ImprovedEO
        
        def decode_wrapper(v, spec_in, max_shift=20):
            from representation.hybrid_pro import decode_hybrid
            return decode_hybrid(self.seed_alignment, v, self.spec, max_shift=max_shift)
        
        def fitness_fn(alignment):
            sp = sp_score(alignment)
            # پاداش اضافی برای همترازی‌های کوتاه‌تر
            length_penalty = len(alignment[0]) / 1000
            return sp - length_penalty, {"sp": sp}
        
        eo = ImprovedEO(decode_wrapper, fitness_fn, self.spec)
        return eo.optimize(
            pop_size=pop_size,
            max_iter=max_iter,
            low=low,
            high=high,
            verbose=False
        )