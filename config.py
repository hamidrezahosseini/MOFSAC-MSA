# config.py
import torch

class Config:
    # تنظیمات پایه
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # تنظیمات پیش‌فرض EO-Pro
    EO_DEFAULT = {
        'pop_size': 50,
        'max_iter': 100,
        'low': -5.0,
        'high': 5.0,
        'eq_pool_size': 5,
        'p_mut': 0.15,
        'p_reseed': 0.02,
        'levy_scale': 0.02,
        'GP_min': 0.5,
        'GP_max': 0.9
    }
    
    # محدوده پارامترهای قابل تنظیم با RL
    EO_PARAM_BOUNDS = {
        'pop_size': (20, 150),
        'max_iter': (30, 300),
        'low': (-10.0, 0.0),
        'high': (0.0, 10.0),
        'eq_pool_size': (2, 12),
        'p_mut': (0.05, 0.4),
        'p_reseed': (0.01, 0.1),
        'levy_scale': (0.01, 0.1)
    }
    
    # تنظیمات SAC
    SAC_CONFIG = {
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'alpha': 0.2,
        'auto_entropy_tuning': True,
        'hidden_size': 256,
        'num_layers': 3
    }
    
    # تنظیمات محیط RL
    RL_ENV_CONFIG = {
        'state_dim': 10,
        'action_dim': 8,          # ۸ پارامتر اصلی (مطابق با environment)
        'max_episode_steps': 1,   # هر اپیزود یک گام (اجرای EO با پارامترهای پیشنهادی)
        'reward_scale': 100.0,
        'penalty_for_invalid': -100.0
    }
    
    # تنظیمات آموزش
    TRAINING_CONFIG = {
        'total_timesteps': 100000,
        'eval_freq': 1000,
        'save_freq': 5000,
        'log_dir': './logs',
        'model_dir': './trained_models'
    }

config = Config()