"""
پایپ‌لاین نهایی بهبودیافته برای همترازی توالی‌های RNA
با حفظ ساختار اصلی و اضافه کردن معیارهای ارزیابی پیشرفته
(اصلاح‌شده برای پردازش تمام فایل‌های Bali2dna_input و ذخیره‌سازی نتایج در Bali2dna_output)
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

# Import ماژول‌های موجود
try:
    from simple_representation import SimpleGapRepresentation
    from advanced_fitness import AdvancedFitness
    from optimized_eo import OptimizedEO
except ImportError:
    # در صورت عدم وجود، جایگزین‌های ساده ایجاد می‌کنیم
    print("⚠️  برخی ماژول‌ها یافت نشدند. از جایگزین‌های ساده استفاده می‌شود.")
    
    class SimpleGapRepresentation:
        def __init__(self, seed_alignment):
            self.seed_alignment = seed_alignment
            self.total_gaps = sum(s.count('-') for s in seed_alignment)
            
        def get_dimension(self):
            return self.total_gaps
            
        def decode(self, vector, max_shift=3):
            # یک decode ساده
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
            
        def optimize(self, pop_size=40, max_iter=50, low=-1.0, high=1.0, seed=None, verbose=True):
            # یک بهینه‌سازی ساده
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
    
    # SP Score
    sp = sp_score(alignment)
    
    # TC Score (ساده)
    perfect_columns = 0
    for col in range(aln_len):
        col_chars = [alignment[i][col] for i in range(n)]
        non_gap_chars = [c for c in col_chars if c != '-']
        if len(non_gap_chars) > 0 and len(set(non_gap_chars)) == 1:
            perfect_columns += 1
    tc_score = perfect_columns / aln_len if aln_len > 0 else 0
    
    # Gap Percentage
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


def run_final_pipeline_improved(fasta_path="input.fasta"):
    """پایپ‌لاین نهایی بهبودیافته با معیارهای ارزیابی پیشرفته"""
    
    print("="*70)
    print("FINAL IMPROVED PIPELINE FOR RNA SEQUENCE ALIGNMENT")
    print("="*70)
    
    # 1. بارگذاری داده‌ها
    print("\n📂 Loading sequences...")
    sequences = read_fasta(fasta_path)[1]
    print(f"   ✓ Loaded {len(sequences)} sequences")
    print(f"   Sequence length range: {min(len(s) for s in sequences)}-{max(len(s) for s in sequences)}")
    
    # 2. روش‌های پایه
    print("\n" + "-"*70)
    print("STEP 1: BASELINE METHODS")
    print("-"*70)
    
    # MAFFT
    print("\n🔧 Running MAFFT...")
    start = time.time()
    aln_mafft = run_mafft(sequences)
    mafft_time = time.time() - start
    sp_mafft = sp_score(aln_mafft)
    print(f"   ✓ MAFFT: SP={sp_mafft:.0f}, Time={mafft_time:.1f}s")
    
    # ClustalW
    print("\n🔧 Running ClustalW...")
    start = time.time()
    aln_clustal = run_clustalw(sequences)
    clustal_time = time.time() - start
    sp_clustal = sp_score(aln_clustal)
    print(f"   ✓ ClustalW: SP={sp_clustal:.0f}, Time={clustal_time:.1f}s")
    
    # انتخاب seed
    if sp_clustal >= sp_mafft:
        seed_alignment = aln_clustal
        seed_sp = sp_clustal
        seed_name = "ClustalW"
    else:
        seed_alignment = aln_mafft
        seed_sp = sp_mafft
        seed_name = "MAFFT"
    
    print(f"\n🎯 Selected seed: {seed_name} (SP={seed_sp:.0f})")
    
    # 3. آماده‌سازی برای EO
    print("\n" + "-"*70)
    print("STEP 2: PREPARING FOR OPTIMIZATION")
    print("-"*70)
    
    print("\n📊 Creating representation...")
    representation = SimpleGapRepresentation(seed_alignment)
    dim = representation.get_dimension()
    print(f"   ✓ Representation dimension: {dim}")
    print(f"   ✓ Total gaps in seed: {representation.total_gaps}")
    
    print("\n⚙️  Setting up fitness function...")
    fitness_calculator = AdvancedFitness()
    
    def decode_fn(vector):
        return representation.decode(vector, max_shift=3)
    
    def fitness_fn(alignment):
        return fitness_calculator.calculate(alignment)
    
    # 4. اجرای EO بهینه‌سازی شده
    print("\n" + "-"*70)
    print("STEP 3: OPTIMIZED EO ALGORITHM")
    print("-"*70)
    
    print("\n🚀 Running Optimized EO...")
    eo = OptimizedEO(decode_fn, fitness_fn, dim)
    
    start = time.time()
    eo_result = eo.optimize(
        pop_size=min(50, max(20, dim * 2)),  # متناسب با بعد
        max_iter=100,
        low=-1.0,
        high=1.0,
        seed=42,
        verbose=True
    )
    eo_time = time.time() - start
    
    aln_eo = eo_result["best_alignment"]
    eo_fitness = eo_result["best_fitness"]
    sp_eo = sp_score(aln_eo)
    
    print(f"\n✅ Optimized EO completed:")
    print(f"   - Final fitness: {eo_fitness:.1f}")
    print(f"   - SP score: {sp_eo:.0f}")
    print(f"   - Improvement from seed: {sp_eo - seed_sp:+.0f}")
    print(f"   - Time: {eo_time:.1f}s")
    
    # 5. اصلاح محلی چندمرحله‌ای
    print("\n" + "-"*70)
    print("STEP 4: MULTI-STAGE LOCAL REFINEMENT")
    print("-"*70)
    
    print("\n🔧 Applying local refinement...")
    
    # Stage 1: Conservative refinement
    print("   Stage 1: Conservative refinement...")
    aln_refined1 = local_refine_alignment(aln_eo, max_iters=50)
    sp_refined1 = sp_score(aln_refined1)
    
    # Stage 2: Aggressive gap shifting
    print("   Stage 2: Aggressive gap optimization...")
    
    def aggressive_gap_optimization(alignment):
        """بهینه‌سازی تهاجمی شکاف‌ها"""
        from copy import deepcopy
        
        aln = [list(seq) for seq in alignment]
        n = len(aln)
        L = len(aln[0])
        
        # شناسایی ستون‌های با شکاف زیاد
        cols_to_remove = []
        for col in range(L):
            gap_count = sum(1 for i in range(n) if aln[i][col] == '-')
            
            # اگر بیش از 50% شکاف باشد، علامت برای حذف
            if gap_count > n // 2:
                cols_to_remove.append(col)
        
        # حذف ستون‌ها از انتها به ابتدا
        for col in reversed(cols_to_remove):
            temp_aln = [row[:col] + row[col+1:] for row in aln]
            temp_aln_str = [''.join(row) for row in temp_aln]
            if sp_score(temp_aln_str) > sp_score([''.join(row) for row in aln]):
                aln = [list(row) for row in temp_aln_str]
                L -= 1
        
        return [''.join(row) for row in aln]
    
    aln_refined2 = aggressive_gap_optimization(aln_refined1)
    sp_refined2 = sp_score(aln_refined2)
    
    # Stage 3: Final touch
    print("   Stage 3: Final optimization...")
    aln_final = local_refine_alignment(aln_refined2, max_iters=30)
    sp_final = sp_score(aln_final)
    
    print(f"\n✅ Refinement results:")
    print(f"   - After stage 1: SP={sp_refined1:.0f}")
    print(f"   - After stage 2: SP={sp_refined2:.0f}")
    print(f"   - Final: SP={sp_final:.0f}")
    
    # 6. محاسبه معیارهای پیشرفته برای همه روش‌ها
    print("\n" + "-"*70)
    print("STEP 5: ADVANCED METRICS CALCULATION")
    print("-"*70)
    
    print("\n📊 Calculating advanced metrics for all methods...")
    
    # محاسبه زمان کل برای روش نهایی
    final_total_time = eo_time + (time.time() - start)
    
    # محاسبه معیارها برای هر روش
    if HAS_ADVANCED_METRICS:
        mafft_metrics = compute_all_metrics(aln_mafft, "MAFFT", mafft_time)
        clustal_metrics = compute_all_metrics(aln_clustal, "ClustalW", clustal_time)
        eo_metrics = compute_all_metrics(aln_eo, "Optimized_EO", eo_time)
        final_metrics = compute_all_metrics(aln_final, "Final_Refined", final_total_time)
    else:
        mafft_metrics = compute_basic_metrics(aln_mafft, "MAFFT", mafft_time)
        clustal_metrics = compute_basic_metrics(aln_clustal, "ClustalW", clustal_time)
        eo_metrics = compute_basic_metrics(aln_eo, "Optimized_EO", eo_time)
        final_metrics = compute_basic_metrics(aln_final, "Final_Refined", final_total_time)
    
    # ذخیره همه معیارها
    all_metrics = {
        "MAFFT": mafft_metrics,
        "ClustalW": clustal_metrics,
        "Optimized_EO": eo_metrics,
        "Final_Refined": final_metrics
    }
    
    # نمایش معیارهای کلیدی
    print(f"\n📋 KEY METRICS SUMMARY:")
    print("-" * 80)
    print(f"{'Method':<20} {'SP Score':<10} {'TC Score':<10} {'CS Score':<10} {'Gap %':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for method_name, metrics in all_metrics.items():
        sp_val = metrics.get('sp_score', 0)
        tc_val = metrics.get('tc_score', 0)
        cs_val = metrics.get('cs_score', metrics.get('tc_score', 0))  # اگر CS نداریم از TC استفاده می‌کنیم
        gap_pct = metrics.get('gap_percentage', 0)
        time_val = metrics.get('execution_time', 0)
        
        print(f"{method_name:<20} {sp_val:<10.0f} {tc_val:<10.3f} {cs_val:<10.3f} {gap_pct:<10.1f} {time_val:<10.2f}")
    
    # 7. ارزیابی و مقایسه جامع
    print("\n" + "-"*70)
    print("STEP 6: COMPREHENSIVE EVALUATION")
    print("-"*70)
    
    # جمع‌آوری نتایج برای مقایسه
    comparison_data = []
    for method_name, metrics in all_metrics.items():
        row = {
            'Method': method_name,
            'SP Score': metrics.get('sp_score', 0),
            'TC Score': metrics.get('tc_score', 0),
            'CS Score': metrics.get('cs_score', metrics.get('tc_score', 0)),
            'Gap %': metrics.get('gap_percentage', 0),
            'Time (s)': metrics.get('execution_time', 0),
            'Conservation': metrics.get('conservation_score', 0),
            'Avg Identity': metrics.get('avg_identity', 0)
        }
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # مرتب‌سازی بر اساس SP Score
    df_comparison = df_comparison.sort_values('SP Score', ascending=False)
    
    print("\n🏆 FINAL RANKING BY SP SCORE:")
    print("-" * 90)
    print(df_comparison[['Method', 'SP Score', 'TC Score', 'CS Score', 'Gap %', 'Time (s)']].to_string(index=False))
    
    # 8. تحلیل نتایج
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)
    
    # بهترین روش بر اساس SP Score
    best_method_sp = df_comparison.iloc[0]['Method']
    best_sp = df_comparison.iloc[0]['SP Score']
    
    # بهترین روش بر اساس Time
    best_method_time = df_comparison.loc[df_comparison['Time (s)'].idxmin()]['Method']
    best_time = df_comparison['Time (s)'].min()
    
    # بهترین روش بر اساس TC Score
    best_method_tc = df_comparison.loc[df_comparison['TC Score'].idxmax()]['Method']
    best_tc = df_comparison['TC Score'].max()
    
    print(f"\n🎯 Best performing method (SP Score): {best_method_sp} (SP={best_sp:.0f})")
    print(f"⚡ Fastest method: {best_method_time} ({best_time:.2f}s)")
    print(f"🏅 Best conserved method (TC Score): {best_method_tc} (TC={best_tc:.3f})")
    
    # مقایسه با baseline
    baseline_sp = max(sp_mafft, sp_clustal)
    baseline_method = "MAFFT" if sp_mafft >= sp_clustal else "ClustalW"
    
    if best_sp > baseline_sp:
        improvement = best_sp - baseline_sp
        percentage = (improvement / baseline_sp) * 100
        print(f"\n✅ SUCCESS: Our best method ({best_method_sp}) beats {baseline_method} by {improvement:.0f} points ({percentage:.1f}%)")
        
        if improvement > 100:
            print("🎉 Excellent improvement achieved!")
        elif improvement > 50:
            print("👏 Very good improvement achieved!")
        elif improvement > 20:
            print("👍 Good improvement achieved!")
        else:
            print("🔍 Small but meaningful improvement achieved!")
    else:
        difference = baseline_sp - best_sp
        print(f"\n❌ Our best method is {difference:.0f} points behind {baseline_method}")
        
        if difference < 10:
            print("⚠️  Performance is very close to baseline")
        else:
            print("🔧 Further optimization needed")
    
    # 9. ایجاد نمودارهای مقایسه
    print("\n" + "-"*70)
    print("STEP 7: VISUALIZATION")
    print("-"*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"results/final_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n📁 Creating experiment directory: {exp_dir}")
    
    # ایجاد نمودارهای مقایسه
    try:
        plots_dir = os.path.join(exp_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # نمودار 1: مقایسه SP Score
        plt.figure(figsize=(10, 6))
        methods = df_comparison['Method']
        sp_scores = df_comparison['SP Score']
        
        colors = ['#FF6B6B' if 'EO' in m or 'Final' in m else '#4ECDC4' if 'MAFFT' in m else '#45B7D1' for m in methods]
        
        bars = plt.bar(methods, sp_scores, color=colors, edgecolor='black')
        plt.title('SP Score Comparison Across Methods', fontsize=14)
        plt.ylabel('SP Score', fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # افزودن مقادیر روی میله‌ها
        for bar, sp in zip(bars, sp_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sp_scores)*0.01,
                    f'{sp:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        sp_plot_path = os.path.join(plots_dir, "sp_score_comparison.png")
        plt.savefig(sp_plot_path, dpi=300)
        print(f"   ✓ SP Score comparison plot saved: {sp_plot_path}")
        
        # نمودار 2: مقایسه TC Score
        plt.figure(figsize=(10, 6))
        tc_scores = df_comparison['TC Score']
        
        bars = plt.bar(methods, tc_scores, color=colors, edgecolor='black')
        plt.title('TC Score Comparison Across Methods', fontsize=14)
        plt.ylabel('TC Score', fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, tc in zip(bars, tc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tc_scores)*0.01,
                    f'{tc:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        tc_plot_path = os.path.join(plots_dir, "tc_score_comparison.png")
        plt.savefig(tc_plot_path, dpi=300)
        print(f"   ✓ TC Score comparison plot saved: {tc_plot_path}")
        
        # نمودار 3: مقایسه زمان اجرا
        plt.figure(figsize=(10, 6))
        times = df_comparison['Time (s)']
        
        bars = plt.bar(methods, times, color=colors, edgecolor='black')
        plt.title('Execution Time Comparison', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, t in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{t:.2f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        time_plot_path = os.path.join(plots_dir, "execution_time_comparison.png")
        plt.savefig(time_plot_path, dpi=300)
        print(f"   ✓ Execution time comparison plot saved: {time_plot_path}")
        
        # نمودار 4: Radar chart (اگر معیارهای کافی وجود دارد)
        if HAS_ADVANCED_METRICS and len(all_metrics) >= 3:
            try:
                # انتخاب 4 معیار برای نمودار رادار
                radar_metrics = ['sp_score', 'tc_score', 'conservation_score', 'avg_identity']
                
                # نرمال‌سازی مقادیر
                normalized_data = {}
                for method_name, metrics in all_metrics.items():
                    normalized = {}
                    for metric in radar_metrics:
                        value = metrics.get(metric, 0)
                        max_val = max([m.get(metric, 0) for m in all_metrics.values()])
                        normalized[metric] = value / max_val if max_val > 0 else 0
                    normalized_data[method_name] = normalized
                
                # ایجاد نمودار رادار
                angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
                angles += angles[:1]
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                
                for idx, (method_name, data) in enumerate(normalized_data.items()):
                    values = [data[metric] for metric in radar_metrics]
                    values += values[:1]
                    
                    ax.plot(angles, values, 'o-', linewidth=2, label=method_name, color=colors[idx % len(colors)])
                    ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels([metric.replace('_', ' ').title() for metric in radar_metrics])
                ax.set_ylim(0, 1)
                ax.set_title('Radar Chart Comparison', size=14, y=1.1)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                
                plt.tight_layout()
                radar_plot_path = os.path.join(plots_dir, "radar_comparison.png")
                plt.savefig(radar_plot_path, dpi=300)
                print(f"   ✓ Radar comparison plot saved: {radar_plot_path}")
                
            except Exception as e:
                print(f"   ⚠️  Could not create radar chart: {e}")
        
        # نمایش نمونه‌ای از نمودارها
        print("\n📈 Visualization completed successfully!")
        
    except Exception as e:
        print(f"   ⚠️  Error creating plots: {e}")
    
    # 10. ذخیره نتایج
    print("\n" + "-"*70)
    print("STEP 8: SAVING RESULTS")
    print("-"*70)
    
    # ذخیره همترازی نهایی
    final_path = os.path.join(exp_dir, "final_alignment.fasta")
    write_fasta([f">seq{i}" for i in range(len(aln_final))], aln_final, final_path)
    
    # ذخیره همه همترازی‌ها
    alignments = {
        "mafft": aln_mafft,
        "clustal": aln_clustal,
        "eo_optimized": aln_eo,
        "final": aln_final
    }
    
    for name, aln in alignments.items():
        path = os.path.join(exp_dir, f"{name}_alignment.fasta")
        write_fasta([f">seq{i}" for i in range(len(aln))], aln, path)
    
    print(f"   ✓ Saved all alignments to FASTA files")
    
    # ذخیره معیارها در JSON
    metrics_path = os.path.join(exp_dir, "all_metrics.json")
    try:
        # تبدیل به فرمت قابل ذخیره در JSON
        serializable_metrics = {}
        for method, metrics in all_metrics.items():
            serializable_metrics[method] = {}
            for key, value in metrics.items():
                if isinstance(value, (np.float32, np.float64)):
                    serializable_metrics[method][key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_metrics[method][key] = int(value)
                else:
                    serializable_metrics[method][key] = value
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Saved all metrics to JSON: {metrics_path}")
    except Exception as e:
        print(f"   ⚠️  Error saving metrics to JSON: {e}")
    
    # ذخیره گزارش متنی
    report_path = os.path.join(exp_dir, "report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("RNA SEQUENCE ALIGNMENT - FINAL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sequences: {len(sequences)}\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 60 + "\n")
        
        for method_name, metrics in all_metrics.items():
            sp_val = metrics.get('sp_score', 0)
            tc_val = metrics.get('tc_score', 0)
            gap_pct = metrics.get('gap_percentage', 0)
            time_val = metrics.get('execution_time', 0)
            
            f.write(f"{method_name:<20}: SP={sp_val:>8.0f}, TC={tc_val:>6.3f}, "
                   f"Gap%={gap_pct:>5.1f}%, Time={time_val:>6.2f}s\n")
        
        f.write(f"\nBest Method (SP Score): {best_method_sp} (SP={best_sp:.0f})\n")
        f.write(f"Fastest Method: {best_method_time} ({best_time:.2f}s)\n")
        f.write(f"Best Conserved Method: {best_method_tc} (TC={best_tc:.3f})\n")
        
        if best_sp > baseline_sp:
            f.write(f"\n✅ Improvement over {baseline_method}: +{best_sp - baseline_sp:.0f} points\n")
        else:
            f.write(f"\n❌ Difference from {baseline_method}: {best_sp - baseline_sp:.0f} points\n")
    
    print(f"   ✓ Saved text report: {report_path}")
    
    # 11. خلاصه نهایی
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n📁 All results saved to: {exp_dir}")
    print(f"📄 Final alignment: {final_path}")
    print(f"📊 Metrics: {metrics_path}")
    print(f"📈 Plots: {plots_dir}/")
    print(f"📋 Report: {report_path}")
    
    # نمایش دستور برای مشاهده نتایج
    print("\n💡 To view results:")
    print(f"   - Open {plots_dir}/ to see comparison plots")
    print(f"   - Check {report_path} for detailed metrics")
    
    # ایجاد دیکشنری results برای سازگاری با کد اصلی
    results = {
        "MAFFT": {"sp": sp_mafft, "time": mafft_time},
        "ClustalW": {"sp": sp_clustal, "time": clustal_time},
        "Optimized_EO": {"sp": sp_eo, "time": eo_time},
        "Final_Refined": {"sp": sp_final, "time": final_total_time}
    }
    
    return aln_final, results, all_metrics, exp_dir


# تابع اصلی اجرا (اصلاح‌شده برای پردازش انبوه)
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BATCH RNA SEQUENCE ALIGNMENT PIPELINE - IMPROVED VERSION")
    print("="*70)
    
    input_dir = "Bali2dna_input"
    output_dir = "Bali2dna_output"
    
    # ایجاد پوشه خروجی در صورت عدم وجود
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isdir(input_dir):
        print(f"\n❌ Directory '{input_dir}' not found. Please create it and add FASTA files.")
        exit(1)
    
    # یافتن تمام فایل‌های FASTA (با پسوندهای رایج)
    fasta_files = []
    for f in os.listdir(input_dir):
        if f.endswith((".fasta", ".fa", ".fna", ".ffn", ".faa", ".frn")):
            fasta_files.append(f)
    
    if not fasta_files:
        print(f"\n⚠️  No FASTA files found in '{input_dir}'. Exiting.")
        exit(0)
        
    print(f"\n📂 Found {len(fasta_files)} FASTA file(s) to process:\n")
    for f in fasta_files:
        print(f"   - {f}")
    
    # پردازش هر فایل
    for i, filename in enumerate(fasta_files, 1):
        filepath = os.path.join(input_dir, filename)
        print("\n" + "#"*70)
        print(f"[{i}/{len(fasta_files)}] PROCESSING: {filename}")
        print("#"*70)
        
        try:
            final_alignment, results, all_metrics, exp_dir = run_final_pipeline_improved(filepath)
            
            # انتقال نتایج به پوشه خروجی با همان نام فایل (بدون پسوند)
            base_name = os.path.splitext(filename)[0]
            target_dir = os.path.join(output_dir, base_name)
            
            # اگر پوشه مقصد وجود داشت، یک عدد به آن اضافه می‌کنیم
            if os.path.exists(target_dir):
                counter = 1
                while os.path.exists(f"{target_dir}_{counter}"):
                    counter += 1
                target_dir = f"{target_dir}_{counter}"
            
            shutil.move(exp_dir, target_dir)
            print(f"   ✅ Results moved to: {target_dir}")
            
        except Exception as e:
            print(f"\n❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETED")
    print("="*70)
    print(f"\n🎉 All results saved in: {output_dir}/")
