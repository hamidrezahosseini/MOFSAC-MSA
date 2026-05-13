"""
پایپ‌لاین نهایی بهبودیافته برای همترازی توالی‌های RNA
با حفظ ساختار اصلی و اضافه کردن معیارهای ارزیابی پیشرفته
(اصلاح‌شده برای پردازش تمام فایل‌های Bali2dna_input و ذخیره‌سازی نتایج در Bali2dna_output)

نسخه مجهز به SAC برای بهینه‌سازی پارامترهای EO
"""
import os
import time
import shutil
import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import توابع پایه
from utils.fasta_io import read_fasta, write_fasta
from tools.external_aligners import run_mafft, run_clustalw
from tools.msa_scoring import sp_score
from tools.local_refinement import local_refine_alignment

# Import ماژول‌های RL (SAC)
from rl.environment import EOEnvironment
from rl.sac_trainer import SACTrainer
# از PrioritizedReplayBuffer یا ReplayBuffer (بسته به نیاز)
from rl.replay_buffer import ReplayBuffer
from config import config  # حتماً config.py در مسیر باشد

# Import ماژول‌های موجود برای پایپ‌لاین قدیمی (اختیاری)
try:
    from simple_representation import SimpleGapRepresentation
    from advanced_fitness import AdvancedFitness
    from optimized_eo import OptimizedEO
except ImportError:
    print("⚠️  برخی ماژول‌ها یافت نشدند. از جایگزین‌های ساده استفاده می‌شود.")
    
    class SimpleGapRepresentation:
        def __init__(self, seed_alignment):
            self.seed_alignment = seed_alignment
            self.total_gaps = sum(s.count('-') for s in seed_alignment)
            
        def get_dimension(self):
            return self.total_gaps
            
        def decode(self, vector, max_shift=3):
            return self.seed_alignment
    
    class AdvancedFitness:
        def calculate(self, alignment):
            score = sp_score(alignment)
            return score, {"sp": score}
    
    class OptimizedEO:
        def __init__(self, decode_fn, fitness_fn, dim):
            self.decode_fn = decode_fn
            self.fitness_fn = fitness_fn
            self.dim = dim
            
        def optimize(self, pop_size=50, max_iter=100, low=-1.0, high=1.0, seed=None, verbose=True):
            print("⚠️  OptimizedEO ساده اجرا می‌شود.")
            return {
                "best_alignment": self.decode_fn(np.zeros(self.dim)),
                "best_fitness": 0,
                "history": []
            }

# Import ماژول‌های جدید برای معیارهای پیشرفته
try:
    from scoring.advanced_metrics import compute_all_metrics
    from visualization import AlignmentVisualizer
    HAS_ADVANCED_METRICS = True
except ImportError:
    print("⚠️  ماژول‌های advanced_metrics یا visualization یافت نشدند.")
    HAS_ADVANCED_METRICS = False


def compute_basic_metrics(alignment, method_name="", exec_time=0):
    """محاسبه معیارهای پایه در صورت عدم وجود ماژول پیشرفته"""
    if not alignment or len(alignment) == 0:
        return {}
    
    n = len(alignment)
    aln_len = len(alignment[0])
    
    sp = sp_score(alignment)
    
    perfect_columns = 0
    for col in range(aln_len):
        col_chars = [alignment[i][col] for i in range(n)]
        non_gap_chars = [c for c in col_chars if c != '-']
        if len(non_gap_chars) > 0 and len(set(non_gap_chars)) == 1:
            perfect_columns += 1
    tc_score = perfect_columns / aln_len if aln_len > 0 else 0
    
    total_gaps = sum(s.count('-') for s in alignment)
    total_chars = n * aln_len
    gap_pct = (total_gaps / total_chars) * 100 if total_chars > 0 else 0
    
    return {
        'method': method_name,
        'sp_score': sp,
        'tc_score': tc_score,
        'gap_percentage': gap_pct,
        'execution_time': exec_time,
        'alignment_length': aln_len,
        'num_sequences': n
    }


# تابع اصلی اجرای دسته‌ای با SAC
def run_batch_with_sac(input_dir="Bali2dna_input", output_dir="Bali2dna_output", total_sac_timesteps=2000):
    """
    پردازش تمام فایل‌های FASTA در input_dir با استفاده از SAC برای بهینه‌سازی پارامترهای EO
    و ذخیره‌ی نتایج در output_dir با همان نام فایل.
    """
    os.makedirs(output_dir, exist_ok=True)

    fasta_files = [f for f in os.listdir(input_dir)
                   if f.endswith((".fasta", ".fa", ".fna", ".ffn", ".faa", ".frn"))]

    if not fasta_files:
        print("No FASTA files found.")
        return

    for i, filename in enumerate(fasta_files, 1):
        filepath = os.path.join(input_dir, filename)
        print(f"\n{'#'*60}")
        print(f"[{i}/{len(fasta_files)}] SAC-EO for {filename}")
        print(f"{'#'*60}")

        try:
            # ۱. ساخت محیط (محیط خودش همترازی اولیه را با MAFFT/ClustalW می‌سازد)
            env = EOEnvironment(fasta_path=filepath)

            # ۲. آموزش SAC
            trainer = SACTrainer(env, config_dict=config.SAC_CONFIG)
            print(f"Starting SAC training with {total_sac_timesteps} timesteps...")
            trainer.train(total_timesteps=total_sac_timesteps)

            # ۳. دریافت بهترین پارامترهای یافت شده در آموزش
            episode_data = env.episode_data
            best_params = episode_data.best_params if episode_data else None
            if best_params is None or not best_params:
                print("⚠️ No best params found, using default.")
                best_params = {
                    'pop_size': 40, 'max_iter': 50, 'low': -1.0, 'high': 1.0,
                    'eq_pool_size': 4, 'p_mut': 0.1, 'p_reseed': 0.1, 'levy_scale': 1.0
                }

            # ۴. اجرای نهایی EO با بهترین پارامترها
            final_sp, eo_info = env._run_eo_with_params(best_params)
            # eo_info['result']['best_alignment'] باید همترازی نهایی را برگرداند
            final_alignment = eo_info['result'].get('best_alignment', env.seed_alignment)

            # ۵. ذخیره‌ی نتایج
            base_name = os.path.splitext(filename)[0]
            res_dir = os.path.join(output_dir, base_name)
            os.makedirs(res_dir, exist_ok=True)

            # ذخیره همترازی نهایی
            aln_path = os.path.join(res_dir, "best_alignment.fasta")
            write_fasta(
                [f">seq{j}" for j in range(len(final_alignment))],
                final_alignment,
                aln_path
            )

            # ذخیره‌ی خلاصه‌ی آموزش و نتایج
            summary = {
                "file": filename,
                "seed_sp": env.seed_sp,
                "final_sp": final_sp,
                "improvement": final_sp - env.seed_sp,
                "best_params": best_params,
                "total_episodes": trainer.total_episodes,
                "total_steps": trainer.total_steps,
                "training_history": {
                    "rewards": trainer.history.get('rewards', [])[-50:],   # ۵۰ آخر
                    "improvements": trainer.history.get('improvements', [])[-50:]
                }
            }
            with open(os.path.join(res_dir, "sac_training_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"  Final SP: {final_sp:.1f} (improvement: {final_sp - env.seed_sp:+.1f})")
            print(f"  Results saved to {res_dir}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue


# نقطه‌ی شروع برنامه
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BATCH RNA ALIGNMENT WITH SAC-EO OPTIMIZATION")
    print("="*70)
    
    # تعداد گام‌های آموزشی SAC (هر چه بیشتر، اپیزود بیشتر و یادگیری بهتر)
    TOTAL_SAC_TIMESTEPS = 2000   # می‌توانید 5000 یا 10000 بگذارید

    run_batch_with_sac(
        input_dir="Bali2dna_input",
        output_dir="Bali2dna_output",
        total_sac_timesteps=TOTAL_SAC_TIMESTEPS
    )
