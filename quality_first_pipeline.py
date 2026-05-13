# quality_first_pipeline.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import time
import json

from config import config
from rl.environment import EOEnvironment
from rl.sac_trainer import SACTrainer
from tools.msa_scoring import sp_score
from representation.hybrid_pro import build_hybrid_spec, decode_hybrid
from eo.equilibrium_optimizer_pro import equilibrium_optimizer_pro
from tools.local_refinement import local_refine_alignment

class QualityFirstPipeline:
    """پایپ‌لاین با اولویت کیفیت"""
    
    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self.results_dir = "quality_first_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def run(self):
        print("="*70)
        print("QUALITY-FIRST RNA ALIGNMENT PIPELINE")
        print("="*70)
        
        # 1. بارگذاری داده‌ها
        from utils.fasta_io import read_fasta
        names, sequences = read_fasta(self.fasta_path)
        print(f"\n1. Loaded {len(sequences)} sequences")
        
        # 2. اجرای همترازی‌های اولیه
        print("\n2. Running initial alignments...")
        from tools.external_aligners import run_mafft, run_clustalw
        
        aln_mafft = run_mafft(sequences)
        sp_mafft = sp_score(aln_mafft)
        
        aln_clustal = run_clustalw(sequences)
        sp_clustal = sp_score(aln_clustal)
        
        print(f"   MAFFT SP: {sp_mafft}")
        print(f"   ClustalW SP: {sp_clustal}")
        
        # انتخاب بهترین
        if sp_clustal >= sp_mafft:
            self.seed_alignment = aln_clustal
            self.seed_sp = sp_clustal
            seed_source = "ClustalW"
        else:
            self.seed_alignment = aln_mafft
            self.seed_sp = sp_mafft
            seed_source = "MAFFT"
            
        print(f"   Selected seed from {seed_source} with SP: {self.seed_sp}")
        
        # 3. آموزش RL با تمرکز بر کیفیت
        print("\n3. Training RL agent (quality-focused)...")
        env = EOEnvironment(sequences=sequences)
        
        # آموزش
        trainer = SACTrainer(env)
        trainer.train(total_timesteps=config.TRAINING_CONFIG['total_timesteps'], 
                     eval_env=env)
        
        # 4. جمع‌آوری بهترین پارامترها از RL
        print("\n4. Collecting best parameters from RL agent...")
        best_params = self._collect_best_params(trainer, env, num_runs=10)
        
        # 5. اجرای نهایی با بهترین پارامترها
        print("\n5. Final optimization with best parameters...")
        final_result = self._run_final_optimization(best_params)
        
        # 6. ذخیره نتایج
        self._save_results(final_result, seed_source)
        
        return final_result
    
    def _collect_best_params(self, trainer, env, num_runs=10):
        """جمع‌آوری بهترین پارامترها از عامل RL"""
        best_params_list = []
        best_sp_list = []
        
        print(f"  Testing {num_runs} parameter sets...")
        
        for i in range(num_runs):
            state = env.reset()
            episode_best_sp = -float('inf')
            episode_best_params = None
            
            while True:
                # انتخاب action
                action = trainer.policy.act(state, deterministic=True)
                
                # تبدیل به پارامتر
                params = env._action_to_params(action)
                
                # اجرای EO-Pro
                sp, info = env._run_eo_with_params(params)
                
                # ذخیره بهترین
                if sp > episode_best_sp:
                    episode_best_sp = sp
                    episode_best_params = params
                
                # مرحله بعد
                next_state, reward, done, info = env.step(action)
                state = next_state
                
                if done:
                    break
            
            best_params_list.append(episode_best_params)
            best_sp_list.append(episode_best_sp)
            
            print(f"    Run {i+1}: SP = {episode_best_sp:.1f}, "
                  f"Improvement = {episode_best_sp - self.seed_sp:.1f}")
        
        # انتخاب بهترین
        best_idx = np.argmax(best_sp_list)
        return best_params_list[best_idx]
    
    def _run_final_optimization(self, params):
        """اجرای بهینه‌سازی نهایی با بهترین پارامترها"""
        # ساخت spec
        spec = build_hybrid_spec(
            self.seed_alignment,
            K_insert=10,  # بزرگتر برای کیفیت بهتر
            K_delete=10,
            M_swaps=8,
            S_segments=10
        )
        
        # توابع
        def decode_wrapper(v, spec_in, max_shift=25):
            return decode_hybrid(self.seed_alignment, v, spec, max_shift=max_shift)
        
        def fitness_fn(alignment):
            score = sp_score(alignment)
            return score, {"sp": score}
        
        print(f"\n  Final optimization parameters:")
        for k, v in params.items():
            if isinstance(v, (np.float32, np.float64)):
                print(f"    {k}: {float(v):.4f}")
            else:
                print(f"    {k}: {v}")
        
        # اجرای نهایی با دقت بالا
        result = equilibrium_optimizer_pro(
            decode_fn=decode_wrapper,
            fitness_fn=fitness_fn,
            spec=spec,
            pop_size=params['pop_size'],
            max_iter=params['max_iter'],
            low=float(params['low']),
            high=float(params['high']),
            eq_pool_size=params['eq_pool_size'],
            p_mut=float(params['p_mut']),
            p_reseed=float(params['p_reseed']),
            levy_scale=float(params['levy_scale']),
            verbose=True,
            seed=config.SEED
        )
        
        return result
    
    def _save_results(self, result, seed_source):
        """ذخیره نتایج"""
        optimized_sp = result['best_fitness']
        optimized_alignment = result['best_alignment']
        
        # اعمال refinement نهایی
        print("\n6. Applying final refinement...")
        refined_alignment = local_refine_alignment(optimized_alignment, max_iters=100)
        refined_sp = sp_score(refined_alignment)
        
        # ذخیره alignments
        def save_fasta(path, alignment, prefix="seq"):
            with open(path, 'w') as f:
                for i, seq in enumerate(alignment):
                    f.write(f">{prefix}{i}\n")
                    for j in range(0, len(seq), 80):
                        f.write(seq[j:j+80] + "\n")
        
        save_fasta(os.path.join(self.results_dir, "seed.fasta"), self.seed_alignment)
        save_fasta(os.path.join(self.results_dir, "optimized.fasta"), optimized_alignment)
        save_fasta(os.path.join(self.results_dir, "refined.fasta"), refined_alignment)
        
        # ذخیره آمار
        stats = {
            "num_sequences": len(self.seed_alignment),
            "alignment_length": len(self.seed_alignment[0]),
            "seed_sp": float(self.seed_sp),
            "seed_source": seed_source,
            "optimized_sp": float(optimized_sp),
            "refined_sp": float(refined_sp),
            "improvement_absolute": float(refined_sp - self.seed_sp),
            "improvement_percentage": float((refined_sp - self.seed_sp) / self.seed_sp * 100),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(self.results_dir, "statistics.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # نمایش نتایج
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"Seed SP ({seed_source}): {self.seed_sp:.1f}")
        print(f"Optimized SP: {optimized_sp:.1f}")
        print(f"Refined SP: {refined_sp:.1f}")
        print(f"Total Improvement: {refined_sp - self.seed_sp:.1f}")
        print(f"Improvement Percentage: {(refined_sp - self.seed_sp)/self.seed_sp*100:.2f}%")
        print(f"\nResults saved to: {self.results_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--training_steps", type=int, default=200, help="Training steps")
    args = parser.parse_args()
    
    # به‌روزرسانی config
    config.TRAINING_CONFIG['total_timesteps'] = args.training_steps
    
    # اجرای پایپ‌لاین
    pipeline = QualityFirstPipeline(args.fasta)
    pipeline.run()

if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    main()