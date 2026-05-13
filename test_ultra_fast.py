# main_rl.py
import os
import argparse
import numpy as np
import torch
from datetime import datetime
import json

# تنظیمات
from config import config

# RL Components
from rl.environment import EOEnvironment
from rl.sac_trainer import SACTrainer

# سایر کامپوننت‌ها
from tools.msa_scoring import sp_score
from tools.local_refinement import local_refine_alignment
from tools.external_aligners import run_mafft, run_clustalw
from utils.fasta_io import read_fasta, write_fasta

def setup_experiment(args):
    """تنظیمات آزمایش"""
    exp_name = args.exp_name or f"eo_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    
    # ایجاد دایرکتوری‌ها
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    # ذخیره تنظیمات
    config_dict = {
        'exp_name': exp_name,
        'fasta_path': args.fasta,
        'training_steps': args.training_steps,
        'config': config.__dict__
    }
    
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Experiment setup complete: {exp_name}")
    print(f"Experiment directory: {exp_dir}")
    
    return exp_dir

def train_rl_agent(fasta_path: str, exp_dir: str, training_steps: int = 10000):
    """آموزش عامل RL"""
    print("\n" + "="*60)
    print("PHASE 1: TRAINING RL AGENT")
    print("="*60)
    
    # ایجاد محیط
    print("Creating RL environment...")
    env = EOEnvironment(fasta_path=fasta_path)
    
    # ایجاد محیط ارزیابی
    eval_env = EOEnvironment(fasta_path=fasta_path)
    
    # ایجاد آموزش‌دهنده
    print("Initializing SAC trainer...")
    trainer = SACTrainer(env)
    
    # آموزش
    print(f"Starting training for {training_steps} steps...")
    trainer.train(training_steps, eval_env)
    
    # ذخیره مدل نهایی
    model_path = os.path.join(exp_dir, "models", "sac_eo_final.pt")
    trainer.save_model(model_path)
    
    return trainer, env


from eo.equilibrium_optimizer_pro import equilibrium_optimizer_pro

def run_optimization_with_rl(trainer, env, exp_dir: str):
    """اجرای بهینه‌سازی با استفاده از عامل آموزش‌دیده"""
    print("\n" + "="*60)
    print("PHASE 2: OPTIMIZATION WITH TRAINED RL AGENT")
    print("="*60)
    
    # بارگذاری توالی‌ها
    sequences = env.sequences
    seed_alignment = env.seed_alignment
    seed_sp = env.seed_sp
    
    print(f"Number of sequences: {len(sequences)}")
    print(f"Seed alignment SP: {seed_sp:.2f}")
    print(f"Seed source: {env.seed_source}")
    
    # جمع‌آوری پارامترهای بهینه از عامل
    print("\nCollecting optimal parameters from RL agent...")
    
    best_params_list = []
    best_sp_list = []
    
    # چندین بار اجرا برای یافتن بهترین پارامترها
    for i in range(5):
        state = env.reset()
        episode_best_sp = 0
        episode_best_params = None
        
        while True:
            # انتخاب action با مدل آموزش‌دیده
            action = trainer.policy.act(state, deterministic=True)
            
            # تبدیل به پارامتر
            params = env._action_to_params(action)
            
            # اجرای EO با این پارامترها
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
        
        print(f"  Run {i+1}: SP = {episode_best_sp:.2f}")
    
    # انتخاب بهترین پارامترها
    best_idx = np.argmax(best_sp_list)
    optimal_params = best_params_list[best_idx]
    optimal_sp = best_sp_list[best_idx]
    
    print(f"\nOptimal parameters found:")
    print(f"  SP improvement: {optimal_sp - seed_sp:.2f}")
    print(f"  Parameters: {optimal_params}")
    
    # اجرای نهایی با بهترین پارامترها
    print("\nRunning final optimization with optimal parameters...")
    
    def decode_wrapper(v, spec_in, max_shift=20):
        from representation.hybrid_pro import decode_hybrid
        return decode_hybrid(seed_alignment, v, env.spec, max_shift=max_shift)
    
    def fitness_fn(alignment):
        score = sp_score(alignment)
        return score, {"sp": score}
    
    final_result = equilibrium_optimizer_pro(
        decode_fn=decode_wrapper,
        fitness_fn=fitness_fn,
        spec=env.spec,
        **optimal_params,
        verbose=True,
        seed=config.SEED
    )
    
    best_alignment = final_result["best_alignment"]
    best_sp = final_result["best_fitness"]
    
    print(f"\nFinal optimization results:")
    print(f"  Seed SP: {seed_sp:.2f}")
    print(f"  Optimized SP: {best_sp:.2f}")
    print(f"  Improvement: {best_sp - seed_sp:.2f}")
    
    # اعمال local refinement
    print("\nApplying local refinement...")
    refined_alignment = local_refine_alignment(best_alignment, max_iters=50)
    refined_sp = sp_score(refined_alignment)
    
    print(f"  After refinement SP: {refined_sp:.2f}")
    print(f"  Total improvement: {refined_sp - seed_sp:.2f}")
    
    # ذخیره نتایج
    save_results(exp_dir, {
        'seed_alignment': seed_alignment,
        'seed_sp': seed_sp,
        'optimized_alignment': best_alignment,
        'optimized_sp': best_sp,
        'refined_alignment': refined_alignment,
        'refined_sp': refined_sp,
        'optimal_params': optimal_params,
        'improvement': refined_sp - seed_sp
    })
    
    return {
        'seed_alignment': seed_alignment,
        'optimized_alignment': best_alignment,
        'refined_alignment': refined_alignment,
        'seed_sp': seed_sp,
        'optimized_sp': best_sp,
        'refined_sp': refined_sp,
        'params': optimal_params
    }

def save_results(exp_dir: str, results: dict):
    """ذخیره نتایج"""
    results_dir = os.path.join(exp_dir, "results")
    
    # ذخیره alignments به صورت FASTA
    write_fasta(
        os.path.join(results_dir, "seed_alignment.fasta"),
        [f">seq{i}" for i in range(len(results['seed_alignment']))],
        results['seed_alignment']
    )
    
    write_fasta(
        os.path.join(results_dir, "optimized_alignment.fasta"),
        [f">seq{i}" for i in range(len(results['optimized_alignment']))],
        results['optimized_alignment']
    )
    
    write_fasta(
        os.path.join(results_dir, "refined_alignment.fasta"),
        [f">seq{i}" for i in range(len(results['refined_alignment']))],
        results['refined_alignment']
    )
    
    # ذخیره آمار
    stats = {
        'seed_sp': float(results['seed_sp']),
        'optimized_sp': float(results['optimized_sp']),
        'refined_sp': float(results['refined_sp']),
        'improvement': float(results['improvement']),
        'optimal_params': results['optimal_params'],
        'timestamp': datetime.now().isoformat()
    }
    
    stats_path = os.path.join(results_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")

def compare_methods(fasta_path: str, exp_dir: str):
    """مقایسه روش‌های مختلف"""
    print("\n" + "="*60)
    print("PHASE 3: COMPARISON OF DIFFERENT METHODS")
    print("="*60)
    
    # بارگذاری توالی‌ها
    _, sequences = read_fasta(fasta_path)
    
    results = {}
    
    # 1. MAFFT
    print("\n1. Running MAFFT...")
    try:
        aln_mafft = run_mafft(sequences)
        sp_mafft = sp_score(aln_mafft)
        results['MAFFT'] = {'sp': sp_mafft, 'alignment': aln_mafft}
        print(f"   SP: {sp_mafft:.2f}")
    except Exception as e:
        print(f"   Error: {e}")
        results['MAFFT'] = {'sp': 0, 'alignment': []}
    
    # 2. ClustalW
    print("\n2. Running ClustalW...")
    try:
        aln_clustal = run_clustalw(sequences)
        sp_clustal = sp_score(aln_clustal)
        results['ClustalW'] = {'sp': sp_clustal, 'alignment': aln_clustal}
        print(f"   SP: {sp_clustal:.2f}")
    except Exception as e:
        print(f"   Error: {e}")
        results['ClustalW'] = {'sp': 0, 'alignment': []}
    
    # 3. EO-Pro با پارامترهای پیش‌فرض
    print("\n3. Running EO-Pro with default parameters...")
    try:
        # ایجاد محیط برای گرفتن seed alignment
        env = EOEnvironment(sequences=sequences)
        
        def decode_wrapper(v, spec_in, max_shift=20):
            from representation.hybrid_pro import decode_hybrid
            return decode_hybrid(env.seed_alignment, v, env.spec, max_shift=max_shift)
        
        def fitness_fn(alignment):
            score = sp_score(alignment)
            return score, {"sp": score}
        
        eo_result = equilibrium_optimizer_pro(
            decode_fn=decode_wrapper,
            fitness_fn=fitness_fn,
            spec=env.spec,
            verbose=False,
            seed=config.SEED
        )
        
        sp_eo = eo_result['best_fitness']
        results['EO-Pro (default)'] = {'sp': sp_eo, 'alignment': eo_result['best_alignment']}
        print(f"   SP: {sp_eo:.2f}")
    except Exception as e:
        print(f"   Error: {e}")
        results['EO-Pro (default)'] = {'sp': 0, 'alignment': []}
    
    # 4. EO-Pro با RL (اگر مدل آموزش‌دیده وجود دارد)
    print("\n4. Running EO-Pro with RL-optimized parameters...")
    rl_model_path = os.path.join(exp_dir, "models", "sac_eo_final.pt")
    
    if os.path.exists(rl_model_path):
        try:
            # بارگذاری نتایج قبلی
            results_path = os.path.join(exp_dir, "results", "statistics.json")
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    rl_results = json.load(f)
                
                results['EO-Pro (RL)'] = {
                    'sp': rl_results['refined_sp'],
                    'alignment': []  # می‌تواند از فایل خوانده شود
                }
                print(f"   SP: {rl_results['refined_sp']:.2f}")
                print(f"   Improvement over seed: {rl_results['improvement']:.2f}")
        except Exception as e:
            print(f"   Error loading RL results: {e}")
            results['EO-Pro (RL)'] = {'sp': 0, 'alignment': []}
    else:
        print("   No trained RL model found")
        results['EO-Pro (RL)'] = {'sp': 0, 'alignment': []}
    
    # نمایش مقایسه
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for method, data in results.items():
        print(f"{method:<20}: {data['sp']:>8.2f}")
    
    # ذخیره مقایسه
    comparison_path = os.path.join(exp_dir, "results", "comparison.json")
    with open(comparison_path, 'w') as f:
        # تبدیل به فرمت قابل ذخیره
        save_data = {}
        for method, data in results.items():
            save_data[method] = {
                'sp': float(data['sp']),
                'alignment_length': len(data['alignment'][0]) if data['alignment'] else 0
            }
        json.dump(save_data, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_path}")
    
    return results

def main():
    """تابع اصلی"""
    parser = argparse.ArgumentParser(description="RNA Sequence Alignment with RL-Optimized EO")
    parser.add_argument("--fasta", type=str, required=True, help="Path to input FASTA file")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--training_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--train_only", action="store_true", help="Train only, don't run optimization")
    parser.add_argument("--compare_only", action="store_true", help="Compare methods only")
    
    args = parser.parse_args()
    
    # تنظیم seed برای تکرارپذیری
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # تنظیم آزمایش
    exp_dir = setup_experiment(args)
    
    if args.compare_only:
        # فقط مقایسه روش‌ها
        compare_methods(args.fasta, exp_dir)
        return
    
    if not args.train_only:
        # آموزش عامل RL
        trainer, env = train_rl_agent(args.fasta, exp_dir, args.training_steps)
        
        # اجرای بهینه‌سازی با عامل آموزش‌دیده
        results = run_optimization_with_rl(trainer, env, exp_dir)
        
        # مقایسه با روش‌های دیگر
        compare_methods(args.fasta, exp_dir)
    else:
        # فقط آموزش
        trainer, _ = train_rl_agent(args.fasta, exp_dir, args.training_steps)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Experiment directory: {exp_dir}")
    print("Check the 'results' folder for outputs and comparisons.")

if __name__ == "__main__":
    # برای اجرا در IDLE، پارامترها را به صورت دستی تنظیم کنید
    import sys
    
    # تنظیم پارامترها به صورت دستی
    class Args:
        fasta = "input.fasta"  # مسیر فایل فستای شما
        exp_name = "idle_experiment"
        training_steps = 60
        train_only = False
        compare_only = False
    
    args = Args()
    
    # تنظیم seed برای تکرارپذیری
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # تنظیم آزمایش
    exp_dir = setup_experiment(args)
    
    if args.compare_only:
        # فقط مقایسه روش‌ها
        compare_methods(args.fasta, exp_dir)
    else:
        if not args.train_only:
            # آموزش عامل RL
            trainer, env = train_rl_agent(args.fasta, exp_dir, args.training_steps)
            
            # اجرای بهینه‌سازی با عامل آموزش‌دیده
            results = run_optimization_with_rl(trainer, env, exp_dir)
            
            # مقایسه با روش‌های دیگر
            compare_methods(args.fasta, exp_dir)
        else:
            # فقط آموزش
            trainer, _ = train_rl_agent(args.fasta, exp_dir, args.training_steps)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Experiment directory: {exp_dir}")
    print("Check the 'results' folder for outputs and comparisons.")
