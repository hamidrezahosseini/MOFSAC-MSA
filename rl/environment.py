# rl/environment.py
import gymnasium as gym  
import time
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from copy import deepcopy

from eo.equilibrium_optimizer_pro import equilibrium_optimizer_pro
from representation.hybrid_pro import build_hybrid_spec, decode_hybrid
from tools.msa_scoring import sp_score
from tools.external_aligners import run_mafft, run_clustalw
from config import config

@dataclass
class EOEpisodeData:
    """ذخیره داده‌های هر اپیزود"""
    sequences: List[str]
    seed_alignment: List[str]
    seed_sp: float
    spec: Dict[str, Any]
    best_params: Dict[str, float] = None
    best_sp: float = 0
    history: List[Dict] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []

class EOEnvironment(gym.Env):
    """محیط RL برای بهینه‌سازی پارامترهای EO"""
    
    def __init__(self, fasta_path: str = None, sequences: List[str] = None):
        super().__init__()
        
        # بارگذاری توالی‌ها
        if sequences:
            self.sequences = sequences
        elif fasta_path:
            from utils.fasta_io import read_fasta
            _, self.sequences = read_fasta(fasta_path)
        else:
            raise ValueError("Either sequences or fasta_path must be provided")
        
        # آماده‌سازی اولیه
        self._prepare_initial_alignment()
        self._build_hybrid_spec()
        
        # تنظیم فضای state و action
        self.state_dim = config.RL_ENV_CONFIG['state_dim']
        self.action_dim = config.RL_ENV_CONFIG['action_dim']
        
        # State: [n_seq, len_aln, gap_ratio, seed_sp, 6 ویژگی آماری دیگر]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        
        # Action: پارامترهای پیوسته نرمال‌شده بین [0, 1]
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.action_dim,), 
            dtype=np.float32
        )
        
        # ذخیره داده‌های اپیزود
        self.episode_data: EOEpisodeData = None
        self.current_step = 0
        self.max_steps = config.RL_ENV_CONFIG['max_episode_steps']
        
        # آمارهای تاریخی
        self.best_global_reward = -np.inf
        self.episode_count = 0
        
    def _prepare_initial_alignment(self):
        """ایجاد همترازی اولیه با MAFFT و ClustalW"""
        print("Running initial alignments...")
        
        # اجرای MAFFT
        try:
            self.aln_mafft = run_mafft(self.sequences)
            self.sp_mafft = sp_score(self.aln_mafft)
        except Exception as e:
            print(f"MAFFT failed: {e}")
            self.aln_mafft = self.sequences  # استفاده از توالی‌های اصلی
            self.sp_mafft = sp_score(self.aln_mafft)
        
        # اجرای ClustalW
        try:
            self.aln_clustal = run_clustalw(self.sequences)
            self.sp_clustal = sp_score(self.aln_clustal)
        except Exception as e:
            print(f"ClustalW failed: {e}")
            self.aln_clustal = self.sequences
            self.sp_clustal = sp_score(self.aln_clustal)
        
        # انتخاب بهترین همترازی اولیه
        if self.sp_clustal >= self.sp_mafft:
            self.seed_alignment = deepcopy(self.aln_clustal)
            self.seed_sp = self.sp_clustal
            self.seed_source = "clustal"
        else:
            self.seed_alignment = deepcopy(self.aln_mafft)
            self.seed_sp = self.sp_mafft
            self.seed_source = "mafft"
            
        print(f"Selected seed from {self.seed_source} with SP: {self.seed_sp}")
        
    def _build_hybrid_spec(self):
        """ساخت مشخصات هیبریدی"""
        self.spec = build_hybrid_spec(
            self.seed_alignment,
            K_insert=8,
            K_delete=8,
            M_swaps=6,
            S_segments=8
        )
        
    def _compute_state_features(self) -> np.ndarray:
        """محاسبه ویژگی‌های state"""
        n_seq = len(self.sequences)
        seq_len = len(self.sequences[0]) if n_seq > 0 else 0
        aln_len = len(self.seed_alignment[0]) if self.seed_alignment else 0
        
        # محاسبه نسبت شکاف‌ها
        total_gaps = sum(s.count('-') for s in self.seed_alignment)
        total_chars = n_seq * aln_len
        gap_ratio = total_gaps / total_chars if total_chars > 0 else 0
        
        # محاسبه تنوع توالی‌ها
        seq_diversity = self._compute_sequence_diversity()
        
        # ویژگی‌های آماری
        gap_distribution = self._compute_gap_distribution()
        
        # State vector
        state = np.zeros(self.state_dim, dtype=np.float32)
        state[0] = n_seq
        state[1] = seq_len
        state[2] = aln_len
        state[3] = gap_ratio
        state[4] = self.seed_sp
        state[5] = seq_diversity
        state[6:10] = gap_distribution[:4]  # 4 ویژگی توزیع شکاف
        
        return state
    
    def _compute_sequence_diversity(self) -> float:
        """محاسبه تنوع توالی‌ها"""
        if len(self.sequences) < 2:
            return 0.0
        
        from scipy.spatial.distance import pdist, squareform
        import numpy as np
        
        # تبدیل توالی‌ها به بردار (کدگذاری ساده)
        encoding = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, '-': 4}
        vectors = []
        for seq in self.sequences:
            vec = [encoding.get(c.upper(), 5) for c in seq]
            vectors.append(vec)
        
        # محاسبه ماتریس فاصله
        try:
            dist_matrix = squareform(pdist(vectors, metric='hamming'))
            diversity = np.mean(dist_matrix)
        except:
            diversity = 0.5  # مقدار پیش‌فرض
            
        return diversity
    
    def _compute_gap_distribution(self) -> np.ndarray:
        """محاسبه توزیع شکاف‌ها"""
        if not self.seed_alignment:
            return np.zeros(4)
        
        n_seq = len(self.seed_alignment)
        aln_len = len(self.seed_alignment[0])
        
        # تعداد شکاف به ازای هر ستون
        gap_counts = []
        for col in range(aln_len):
            col_gaps = sum(1 for seq in self.seed_alignment if seq[col] == '-')
            gap_counts.append(col_gaps / n_seq)  # نرمال‌سازی
        
        if not gap_counts:
            return np.zeros(4)
        
        gap_counts = np.array(gap_counts)
        
        # ویژگی‌های آماری
        return np.array([
            np.mean(gap_counts),      # میانگین
            np.std(gap_counts),       # انحراف معیار
            np.max(gap_counts),       # حداکثر
            np.sum(gap_counts > 0.5) / aln_len  # نسبت ستون‌هایی با بیش از 50% شکاف
        ])
    
    def _action_to_params(self, action: np.ndarray) -> Dict[str, float]:
        """تبدیل action به پارامترهای EO"""
        bounds = config.EO_PARAM_BOUNDS
        params = {}
        
        # پارامترهای اصلی
        params['pop_size'] = int(bounds['pop_size'][0] + 
                               action[0] * (bounds['pop_size'][1] - bounds['pop_size'][0]))
        
        params['max_iter'] = int(bounds['max_iter'][0] + 
                               action[1] * (bounds['max_iter'][1] - bounds['max_iter'][0]))
        
        params['low'] = bounds['low'][0] + action[2] * (bounds['low'][1] - bounds['low'][0])
        params['high'] = bounds['high'][0] + action[3] * (bounds['high'][1] - bounds['high'][0])
        
        params['eq_pool_size'] = int(bounds['eq_pool_size'][0] + 
                                   action[4] * (bounds['eq_pool_size'][1] - bounds['eq_pool_size'][0]))
        
        # پارامترهای جهش
        params['p_mut'] = bounds['p_mut'][0] + action[5] * (bounds['p_mut'][1] - bounds['p_mut'][0])
        params['p_reseed'] = bounds['p_reseed'][0] + action[6] * (bounds['p_reseed'][1] - bounds['p_reseed'][0])
        params['levy_scale'] = bounds['levy_scale'][0] + action[7] * (bounds['levy_scale'][1] - bounds['levy_scale'][0])
        
        # پارامترهای کنترل (غیرفعال)
        #params['a1'] = bounds['a1'][0] + action[8] * (bounds['a1'][1] - bounds['a1'][0])
        #params['a2'] = bounds['a2'][0] + action[9] * (bounds['a2'][1] - bounds['a2'][0])
        
        return params
    
    def _run_eo_with_params(self, params: Dict) -> Tuple[float, Dict]:
        try:
            def decode_wrapper(v, spec_in, max_shift=20):
                return decode_hybrid(self.seed_alignment, v, self.spec, max_shift=max_shift)
            
            def fitness_fn(alignment):
                score = sp_score(alignment)
                return score, {"sp": score}
            
            # محدود کردن موقت max_iter برای سرعت (قابل تغییر)
            test_params = params.copy()
            test_params['max_iter'] = min(5, test_params['max_iter'])   # فقط 5 تکرار برای تست
            
            print(f"\n  Running EO-Pro with params: pop_size={test_params['pop_size']}, "
                  f"max_iter={test_params['max_iter']}")
            
            result = equilibrium_optimizer_pro(
                decode_fn=decode_wrapper,
                fitness_fn=fitness_fn,
                spec=self.spec,
                **test_params,
                verbose=False,
                seed=config.SEED
            )
            
            best_sp = result['best_fitness']
            improvement = best_sp - self.seed_sp
            
            print(f"  Result: SP={best_sp:.1f}, Improvement={improvement:.1f}")
            
            return best_sp, {
                'result': result,
                'params': test_params,
                'best_sp': best_sp,
                'improvement': improvement
            }
        except Exception as e:
            print(f"EO execution error: {e}")
            import traceback
            traceback.print_exc()
            return self.seed_sp, {'error': str(e)}
    
    def reset(self) -> np.ndarray:
        """بازنشانی محیط برای شروع اپیزود جدید"""
        self.episode_count += 1
        self.current_step = 0
        
        # ایجاد داده‌های اپیزود جدید
        self.episode_data = EOEpisodeData(
            sequences=self.sequences,
            seed_alignment=deepcopy(self.seed_alignment),
            seed_sp=self.seed_sp,
            spec=self.spec
        )
        
        # محاسبه state اولیه
        state = self._compute_state_features()
        
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """انجام یک مرحله در محیط"""
        self.current_step += 1
        
        # تبدیل action به پارامترها
        params = self._action_to_params(action)
        
        # اجرای EO با پارامترها
        best_sp, info = self._run_eo_with_params(params)
        
        # محاسبه reward
        reward = self._compute_reward(best_sp, info)
        
        # بررسی پایان اپیزود
        done = self.current_step >= self.max_steps
        
        # ذخیره تاریخچه
        self.episode_data.history.append({
            'step': self.current_step,
            'action': action,
            'params': params,
            'best_sp': best_sp,
            'reward': reward,
            'info': info
        })
        
        # به‌روزرسانی بهترین نتایج
        if best_sp > self.episode_data.best_sp:
            self.episode_data.best_sp = best_sp
            self.episode_data.best_params = params
        
        # محاسبه state جدید
        next_state = self._compute_state_features()
        
        # اطلاعات اضافی
        info.update({
            'episode': self.episode_count,
            'step': self.current_step,
            'best_sp': best_sp,
            'seed_sp': self.seed_sp,
            'improvement': best_sp - self.seed_sp,
            'is_success': best_sp > self.seed_sp
        })
        
        return next_state, reward, done, info
    
    def _compute_reward(self, best_sp: float, info: Dict) -> float:
        improvement = best_sp - self.seed_sp
        if improvement > 0:
            reward = improvement * 0.01  
        else:
            reward = improvement * 0.001
        reward = max(-10, min(100, reward))
        return reward
        
    def render(self, mode='human'):
        """نمایش وضعیت محیط"""
        if mode == 'human':
            print(f"\n=== Episode {self.episode_count}, Step {self.current_step} ===")
            if self.episode_data:
                print(f"Best SP this episode: {self.episode_data.best_sp:.2f}")
                print(f"Improvement: {self.episode_data.best_sp - self.seed_sp:.2f}")
                if self.episode_data.best_params:
                    print(f"Best params: {self.episode_data.best_params}")
    
    def close(self):
        """بستن محیط"""
        pass
    
    def get_episode_summary(self) -> Dict:
        """دریافت خلاصه اپیزود"""
        if not self.episode_data:
            return {}
        
        return {
            'episode': self.episode_count,
            'seed_sp': self.episode_data.seed_sp,
            'best_sp': self.episode_data.best_sp,
            'improvement': self.episode_data.best_sp - self.episode_data.seed_sp,
            'improvement_ratio': (self.episode_data.best_sp - self.episode_data.seed_sp) / 
                               (self.episode_data.seed_sp + 1e-8),
            'best_params': self.episode_data.best_params,
            'num_steps': len(self.episode_data.history),
            'seed_source': getattr(self, 'seed_source', 'unknown')
        }