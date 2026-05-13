# run_advanced_pipeline.py
import os
import time
import sys
from datetime import datetime

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_advanced_pipeline(fasta_path="input.fasta"):
    """پایپ‌لاین پیشرفته با استراتژی‌های چندگانه"""
    
    print("="*70)
    print("ADVANCED RNA SEQUENCE ALIGNMENT PIPELINE")
    print("="*70)
    
    # 1. بارگذاری داده‌ها
    from utils.fasta_io import read_fasta
    
    print(f"Loading sequences from {fasta_path}...")
    sequences = read_fasta(fasta_path)[1]
    print(f"✓ Loaded {len(sequences)} sequences")
    
    # 2. اجرای روش‌های پایه
    from tools.external_aligners import run_mafft, run_clustalw
    from tools.msa_scoring import sp_score
    
    print("\n" + "="*70)
    print("STEP 1: BASELINE METHODS")
    print("="*70)
    
    results = {}
    
    # MAFFT
    try:
        print("\nRunning MAFFT...")
        start = time.time()
        aln_mafft = run_mafft(sequences)
        mafft_time = time.time() - start
        sp_mafft = sp_score(aln_mafft)
        results['MAFFT'] = {'sp': sp_mafft, 'time': mafft_time, 'alignment': aln_mafft}
        print(f"✓ SP={sp_mafft:.0f}, Time={mafft_time:.1f}s")
    except Exception as e:
        print(f"✗ MAFFT failed: {e}")
        return None
    
    # ClustalW
    try:
        print("\nRunning ClustalW...")
        start = time.time()
        aln_clustal = run_clustalw(sequences)
        clustal_time = time.time() - start
        sp_clustal = sp_score(aln_clustal)
        results['ClustalW'] = {'sp': sp_clustal, 'time': clustal_time, 'alignment': aln_clustal}
        print(f"✓ SP={sp_clustal:.0f}, Time={clustal_time:.1f}s")
    except Exception as e:
        print(f"✗ ClustalW failed: {e}")
        return None
    
    # 3. انتخاب seed
    seed_alignment = aln_clustal if sp_clustal >= sp_mafft else aln_mafft
    seed_sp = max(sp_clustal, sp_mafft)
    seed_name = "ClustalW" if sp_clustal >= sp_mafft else "MAFFT"
    
    print(f"\nSelected seed: {seed_name} (SP={seed_sp:.0f})")
    
    # 4. اجرای EO پیشرفته با fitness فازی
    print("\n" + "="*70)
    print("STEP 2: ADVANCED EO WITH FUZZY FITNESS")
    print("="*70)
    
    try:
        from fuzzy_advanced import AdvancedFuzzyFitness
        from advanced_eo import AdvancedEO
        from representation.hybrid_pro import build_hybrid_spec, decode_hybrid
        
        print("\nBuilding hybrid representation...")
        spec = build_hybrid_spec(
            seed_alignment,
            K_insert=5,
            K_delete=5,
            M_swaps=4,
            S_segments=5
        )
        
        # تابع fitness پیشرفته
        fuzzy_fitness = AdvancedFuzzyFitness()
        
        def fitness_wrapper(alignment):
            return fuzzy_fitness.calculate(alignment)
        
        def decode_wrapper(v, spec_in, max_shift=20):
            return decode_hybrid(seed_alignment, v, spec, max_shift)
        
        # ایجاد و اجرای EO پیشرفته
        eo = AdvancedEO(decode_wrapper, fitness_wrapper, spec)
        
        print("\nRunning Advanced EO optimization (hybrid strategy)...")
        start = time.time()
        eo_result = eo.optimize(
            pop_size=50,
            max_iter=80,
            low=-4.0,
            high=4.0,
            strategy='hybrid',
            verbose=True,
            seed=42
        )
        eo_time = time.time() - start
        
        aln_eo = eo_result["best_alignment"]
        sp_eo = sp_score(aln_eo)
        results['Advanced_EO'] = {'sp': sp_eo, 'time': eo_time, 'alignment': aln_eo}
        
        print(f"\n✓ Advanced EO: SP={sp_eo:.0f}, Time={eo_time:.1f}s")
        print(f"  Improvement from seed: {sp_eo - seed_sp:+.0f}")
        
    except Exception as e:
        print(f"✗ Advanced EO failed: {e}")
        import traceback
        traceback.print_exc()
        aln_eo = seed_alignment
        sp_eo = seed_sp
    
    # 5. استراتژی Ensemble
    print("\n" + "="*70)
    print("STEP 3: ENSEMBLE STRATEGY")
    print("="*70)
    
    try:
        from ensemble_strategy import EnsembleAlignment
        
        ensemble = EnsembleAlignment(sequences)
        
        print("\nCreating ensemble alignment...")
        start = time.time()
        
        # ترکیب همه همترازی‌ها
        all_alignments = [
            aln_mafft,
            aln_clustal,
            aln_eo
        ]
        
        aln_ensemble = ensemble.hybrid_ensemble(
            aln_mafft, aln_clustal, aln_eo, aln_eo
        )
        
        # اگر ensemble موفق نبود، از column selection استفاده کن
        if not aln_ensemble:
            print("Using column selection strategy...")
            aln_ensemble = ensemble.column_selection(all_alignments)
        
        ensemble_time = time.time() - start
        sp_ensemble = sp_score(aln_ensemble)
        results['Ensemble'] = {'sp': sp_ensemble, 'time': ensemble_time, 'alignment': aln_ensemble}
        
        print(f"✓ Ensemble: SP={sp_ensemble:.0f}, Time={ensemble_time:.1f}s")
        
    except Exception as e:
        print(f"✗ Ensemble failed: {e}")
        aln_ensemble = aln_eo
        sp_ensemble = sp_eo
    
    # 6. اصلاح نهایی پیشرفته
    print("\n" + "="*70)
    print("STEP 4: ADVANCED REFINEMENT")
    print("="*70)
    
    try:
        from tools.local_refinement import local_refine_alignment
        
        print("\nApplying advanced refinement...")
        start = time.time()
        
        # چندین مرحله refinement با پارامترهای مختلف
        aln_refined = aln_ensemble
        
        # Stage 1: Conservative refinement
        aln_refined = local_refine_alignment(aln_refined, max_iters=30)
        
        # Stage 2: Aggressive refinement
        for _ in range(3):
            temp_refined = local_refine_alignment(aln_refined, max_iters=10)
            if sp_score(temp_refined) > sp_score(aln_refined):
                aln_refined = temp_refined
        
        refine_time = time.time() - start
        sp_final = sp_score(aln_refined)
        results['Final'] = {'sp': sp_final, 'time': refine_time, 'alignment': aln_refined}
        
        print(f"✓ Final refinement: SP={sp_final:.0f}, Time={refine_time:.1f}s")
        
    except Exception as e:
        print(f"✗ Refinement failed: {e}")
        aln_refined = aln_ensemble
        sp_final = sp_ensemble
    
    # 7. نمایش نتایج نهایی
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    # مرتب‌سازی بر اساس SP
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sp'], reverse=True)
    
    print("\nRanking of Methods:")
    print("-" * 80)
    print(f"{'Method':<20} {'SP Score':>12} {'Time(s)':>10} {'Improvement':>15} {'Status':>10}")
    print("-" * 80)
    
    for i, (method, data) in enumerate(sorted_results):
        improvement = data['sp'] - seed_sp
        status = "✓ BETTER" if improvement > 0 else "✗ WORSE" if improvement < 0 else "= SAME"
        
        print(f"{i+1:>2}. {method:<18} {data['sp']:>12.0f} {data['time']:>10.1f} "
              f"{improvement:>+15.0f} {status:>10}")
    
    # 8. آنالیز آماری
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)
    
    best_method = sorted_results[0][0]
    best_sp = sorted_results[0][1]['sp']
    
    print(f"\n🏆 Best Method: {best_method} (SP={best_sp:.0f})")
    
    # مقایسه با روش‌های پایه
    print("\nComparison with Baseline Methods:")
    print("-" * 50)
    
    baseline_sp = max(sp_mafft, sp_clustal)
    baseline_method = "MAFFT" if sp_mafft >= sp_clustal else "ClustalW"
    
    improvement_over_baseline = best_sp - baseline_sp
    improvement_percentage = (improvement_over_baseline / baseline_sp) * 100
    
    if improvement_over_baseline > 0:
        print(f"✓ Our method beats {baseline_method} by {improvement_over_baseline:.0f} points "
              f"({improvement_percentage:.2f}%)")
        
        if improvement_over_baseline > 50:
            print(f"🎉 Significant improvement achieved!")
        elif improvement_over_baseline > 20:
            print(f"👏 Good improvement achieved!")
        else:
            print(f"👍 Small but meaningful improvement achieved!")
    else:
        print(f"✗ Our method is worse than {baseline_method} by {abs(improvement_over_baseline):.0f} points")
        
        if abs(improvement_over_baseline) < 10:
            print(f"⚠️  Performance is comparable to baseline")
        else:
            print(f"🔧 Needs further optimization")
    
    # 9. ذخیره نتایج
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/advanced_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    from utils.fasta_io import write_fasta
    
    # ذخیره همترازی نهایی
    final_path = os.path.join(exp_dir, "final_alignment.fasta")
    write_fasta(
        [f">seq{i}" for i in range(len(aln_refined))],
        aln_refined,
        final_path
    )
    
    # ذخیره همه همترازی‌ها
    for method, data in results.items():
        method_path = os.path.join(exp_dir, f"{method.lower()}_alignment.fasta")
        write_fasta(
            [f">seq{i}" for i in range(len(data['alignment']))],
            data['alignment'],
            method_path
        )
    
    # ذخیره گزارش کامل
    report_path = os.path.join(exp_dir, "full_report.txt")
    with open(report_path, 'w') as f:
        f.write("ADVANCED RNA SEQUENCE ALIGNMENT REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sequences: {len(sequences)}\n")
        f.write(f"Input file: {fasta_path}\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 60 + "\n")
        for method, data in results.items():
            f.write(f"{method:<15}: SP={data['sp']:>8.0f}, Time={data['time']:>6.1f}s\n")
        
        f.write(f"\nBest Method: {best_method} (SP={best_sp:.0f})\n")
        f.write(f"Improvement over baseline: {improvement_over_baseline:+.0f} points ({improvement_percentage:+.2f}%)\n")
    
    print(f"\n✓ All results saved to: {exp_dir}")
    print(f"✓ Final alignment: {final_path}")
    print(f"✓ Full report: {report_path}")
    
    return aln_refined, results

def main():
    """تابع اصلی اجرا"""
    
    # بررسی وجود فایل ورودی
    if not os.path.exists("input.fasta"):
        print("Error: input.fasta not found!")
        print("Please create an input.fasta file or specify path.")
        return
    
    try:
        aln_final, results = run_advanced_pipeline("input.fasta")
        
        # نمایش پیام پایانی
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        best_method = max(results.items(), key=lambda x: x[1]['sp'])[0]
        best_sp = results[best_method]['sp']
        
        print(f"\n🎯 Best SP Score Achieved: {best_sp:.0f}")
        print(f"🏆 Best Method: {best_method}")
        print(f"\nCheck the experiment folder for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()