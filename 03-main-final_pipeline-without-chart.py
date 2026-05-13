# final_pipeline.py
import os
import time
import numpy as np
from datetime import datetime

# Import توابع
from utils.fasta_io import read_fasta, write_fasta
from tools.external_aligners import run_mafft, run_clustalw
from tools.msa_scoring import sp_score
from tools.local_refinement import local_refine_alignment

# Import ماژول‌های جدید
from simple_representation import SimpleGapRepresentation
from advanced_fitness import AdvancedFitness
from optimized_eo import OptimizedEO


def run_final_pipeline(fasta_path="input.fasta"):
    """پایپ‌لاین نهایی بهبودیافته"""
    
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
    
    # Stage 2: Aggressive gap shifting (تابع کمکی)
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
        
        # حذف ستون‌ها از انتها به ابتدا (برای جلوگیری از تغییر اندیس)
        for col in reversed(cols_to_remove):
            # بررسی اگر حذف این ستون امتیاز را بهبود می‌دهد
            temp_aln = [row[:col] + row[col+1:] for row in aln]
            temp_aln_str = [''.join(row) for row in temp_aln]
            if sp_score(temp_aln_str) > sp_score([''.join(row) for row in aln]):
                aln = [list(row) for row in temp_aln_str]
                L -= 1
        
        return [''.join(row) for row in aln]
    
    print("   Stage 2: Aggressive gap optimization...")
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
    
    # 6. ارزیابی و مقایسه
    print("\n" + "-"*70)
    print("STEP 5: EVALUATION AND COMPARISON")
    print("-"*70)
    
    # جمع‌آوری همه نتایج
    results = {
        "MAFFT": {"sp": sp_mafft, "time": mafft_time},
        "ClustalW": {"sp": sp_clustal, "time": clustal_time},
        "Optimized_EO": {"sp": sp_eo, "time": eo_time},
        "Final_Refined": {"sp": sp_final, "time": eo_time + (time.time() - start)}
    }
    
    # مرتب‌سازی بر اساس SP
    sorted_results = sorted(results.items(), key=lambda x: x[1]["sp"], reverse=True)
    
    print("\n🏆 FINAL RANKING:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Method':<20} {'SP Score':<12} {'Time(s)':<10} {'Improvement':<15}")
    print("-" * 60)
    
    for rank, (method, data) in enumerate(sorted_results, 1):
        improvement = data["sp"] - seed_sp
        improvement_str = f"{improvement:+.0f}"
        
        if improvement > 0:
            improvement_str = f"↑ {improvement_str}"
        elif improvement < 0:
            improvement_str = f"↓ {improvement_str}"
        else:
            improvement_str = "= 0"
        
        print(f"{rank:<6} {method:<20} {data['sp']:<12.0f} {data['time']:<10.1f} {improvement_str:<15}")
    
    # 7. تحلیل نتایج
    print("\n" + "-"*70)
    print("ANALYSIS")
    print("-"*70)
    
    best_method = sorted_results[0][0]
    best_sp = sorted_results[0][1]["sp"]
    
    print(f"\n🎯 Best performing method: {best_method}")
    print(f"📈 Best SP score achieved: {best_sp:.0f}")
    
    # مقایسه با baseline
    baseline_sp = max(sp_mafft, sp_clustal)
    baseline_method = "MAFFT" if sp_mafft >= sp_clustal else "ClustalW"
    
    if best_sp > baseline_sp:
        improvement = best_sp - baseline_sp
        percentage = (improvement / baseline_sp) * 100
        print(f"✅ SUCCESS: Our method beats {baseline_method} by {improvement:.0f} points ({percentage:.1f}%)")
        
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
        print(f"❌ Our method is {difference:.0f} points behind {baseline_method}")
        
        if difference < 10:
            print("⚠️  Performance is very close to baseline")
        else:
            print("🔧 Further optimization needed")
    
    # 8. ذخیره نتایج
    print("\n" + "-"*70)
    print("SAVING RESULTS")
    print("-"*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"results/final_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
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
    
    # ذخیره گزارش
    report_path = os.path.join(exp_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("RNA SEQUENCE ALIGNMENT - FINAL RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sequences: {len(sequences)}\n\n")
        
        f.write("RESULTS:\n")
        f.write("-" * 60 + "\n")
        for method, data in results.items():
            f.write(f"{method:<20}: SP={data['sp']:>8.0f}, Time={data['time']:>6.1f}s\n")
        
        f.write(f"\nBest Method: {best_method} (SP={best_sp:.0f})\n")
        
        if best_sp > baseline_sp:
            f.write(f"Improvement over {baseline_method}: +{best_sp - baseline_sp:.0f} points\n")
        else:
            f.write(f"Difference from {baseline_method}: {best_sp - baseline_sp:.0f} points\n")
    
    print(f"\n✅ Results saved to: {exp_dir}")
    print(f"📄 Final alignment: {final_path}")
    print(f"📊 Full report: {report_path}")
    
    return aln_final, results

# تابع اصلی اجرا
if __name__ == "__main__":
    if not os.path.exists("input.fasta"):
        print("Creating sample input.fasta...")
        sample_seqs = [
            "AUCGUAUCGUAUCGUAUCGUAUCGU",
            "AUCGUAUCG-AUCGUAUCGUAUCGU",
            "AUCG-AUCGUAUCGUAUCGUAUCGU",
            "AUCGUAUCGUAUCG-AUCGUAUCGU",
            "AUCGUAUCGUAUCGUAUCG-AUCGU",
            "AUCGUAUCGUAUCGUAUCGUAUCG-",
            "-UCGUAUCGUAUCGUAUCGUAUCGU",
            "AUCGUAUCGUAUCGUAUCGUAUCGA"
        ]
        
        with open("input.fasta", "w") as f:
            for i, seq in enumerate(sample_seqs):
                f.write(f">seq_{i+1}\n{seq}\n")
        print("Sample file created.")
    
    # اجرای پایپ‌لاین
    try:
        final_alignment, results = run_final_pipeline()
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETED!")
        print("="*70)
        
        best_method = max(results.items(), key=lambda x: x[1]["sp"])[0]
        best_sp = results[best_method]["sp"]
        
        print(f"\n🎉 Final SP Score: {best_sp:.0f}")
        print(f"🏆 Best Method: {best_method}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
