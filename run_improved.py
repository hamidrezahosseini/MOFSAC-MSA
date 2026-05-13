# run_improved.py
import os
import time
import sys
from datetime import datetime

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_improved_pipeline(fasta_path="input.fasta"):
    """پایپ‌لاین بهبودیافته"""
    
    print("="*60)
    print("IMPROVED RNA SEQUENCE ALIGNMENT PIPELINE")
    print("="*60)
    
    # 1. بارگذاری داده‌ها
    from utils.fasta_io import read_fasta
    
    print(f"Loading sequences from {fasta_path}...")
    sequences = read_fasta(fasta_path)[1]
    print(f"✓ Loaded {len(sequences)} sequences")
    
    # 2. اجرای روش‌های پایه
    from tools.external_aligners import run_mafft, run_clustalw
    from tools.msa_scoring import sp_score
    
    print("\n1. Running baseline methods...")
    
    # MAFFT
    try:
        start = time.time()
        aln_mafft = run_mafft(sequences)
        mafft_time = time.time() - start
        sp_mafft = sp_score(aln_mafft)
        print(f"   ✓ MAFFT: SP={sp_mafft:.0f}, Time={mafft_time:.1f}s")
    except Exception as e:
        print(f"   ✗ MAFFT failed: {e}")
        aln_mafft = sequences
        sp_mafft = sp_score(aln_mafft)
    
    # ClustalW
    try:
        start = time.time()
        aln_clustal = run_clustalw(sequences)
        clustal_time = time.time() - start
        sp_clustal = sp_score(aln_clustal)
        print(f"   ✓ ClustalW: SP={sp_clustal:.0f}, Time={clustal_time:.1f}s")
    except Exception as e:
        print(f"   ✗ ClustalW failed: {e}")
        aln_clustal = sequences
        sp_clustal = sp_score(aln_clustal)
    
    # 3. انتخاب بهترین همترازی اولیه
    if sp_clustal >= sp_mafft:
        seed = aln_clustal
        seed_sp = sp_clustal
        seed_name = "clustalw"
    else:
        seed = aln_mafft
        seed_sp = sp_mafft
        seed_name = "mafft"
    
    print(f"\n2. Using {seed_name} as seed (SP={seed_sp:.0f})")
    
    # 4. اجرای استراتژی ترکیبی
    print("\n3. Running hybrid strategy...")
    try:
        from hybrid_strategy import HybridStrategy
        
        start = time.time()
        hybrid = HybridStrategy(sequences)
        aln_hybrid = hybrid.align()
        hybrid_time = time.time() - start
        sp_hybrid = sp_score(aln_hybrid)
        
        improvement = sp_hybrid - seed_sp
        print(f"   ✓ Hybrid: SP={sp_hybrid:.0f}, Improvement={improvement:+.0f}, Time={hybrid_time:.1f}s")
    except Exception as e:
        print(f"   ✗ Hybrid strategy failed: {e}")
        # استفاده از بهترین روش پایه به عنوان fallback
        aln_hybrid = seed
        sp_hybrid = seed_sp
        print(f"   Using {seed_name} as fallback")
    
    # 5. اجرای EO-Pro با پارامترهای بهینه
    print("\n4. Running optimized EO-Pro...")
    try:
        from representation.hybrid_pro import build_hybrid_spec, decode_hybrid
        from eo.equilibrium_optimizer_pro import equilibrium_optimizer_pro
        
        spec = build_hybrid_spec(
            seed,
            K_insert=4,
            K_delete=4,
            M_swaps=3,
            S_segments=4
        )
        
        def fitness_fn(alignment):
            sp = sp_score(alignment)
            return sp, {"sp": sp}
        
        def decode_wrapper(v, spec_in, max_shift=20):
            return decode_hybrid(seed, v, spec, max_shift)
        
        start = time.time()
        result = equilibrium_optimizer_pro(
            decode_fn=decode_wrapper,
            fitness_fn=fitness_fn,
            spec=spec,
            pop_size=40,
            max_iter=60,
            low=-3.0,
            high=3.0,
            eq_pool_size=4,
            p_mut=0.1,
            p_reseed=0.02,
            levy_scale=0.01,
            verbose=False
        )
        eo_time = time.time() - start
        
        aln_eo = result["best_alignment"]
        sp_eo = sp_score(aln_eo)
        
        print(f"   ✓ Improved EO: SP={sp_eo:.0f}, Time={eo_time:.1f}s")
    except Exception as e:
        print(f"   ✗ EO-Pro failed: {e}")
        aln_eo = seed
        sp_eo = seed_sp
    
    # 6. اعمال اصلاح نهایی
    print("\n5. Applying final refinement...")
    try:
        from tools.local_refinement import local_refine_alignment
        
        # انتخاب بهترین همترازی تا این مرحله
        best_aln = aln_hybrid if sp_hybrid >= sp_eo else aln_eo
        best_sp = max(sp_hybrid, sp_eo)
        
        start = time.time()
        aln_final = local_refine_alignment(best_aln, max_iters=50)
        refine_time = time.time() - start
        sp_final = sp_score(aln_final)
        
        print(f"   ✓ Final refinement: SP={sp_final:.0f}, Time={refine_time:.1f}s")
    except Exception as e:
        print(f"   ✗ Refinement failed: {e}")
        # انتخاب بهترین همترازی بدون refinement
        if sp_hybrid >= sp_eo:
            aln_final = aln_hybrid
            sp_final = sp_hybrid
        else:
            aln_final = aln_eo
            sp_final = sp_eo
    
    # 7. نمایش نتایج
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    results = {
        "MAFFT": sp_mafft,
        "ClustalW": sp_clustal,
        "Hybrid": sp_hybrid,
        "Improved EO": sp_eo,
        "Final": sp_final
    }
    
    # ترتیب بر اساس امتیاز
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nPerformance Comparison:")
    print("-" * 40)
    for method, score in sorted_results:
        improvement_from_mafft = score - sp_mafft
        improvement_from_clustal = score - sp_clustal
        print(f"{method:<15}: {score:>8.0f} | "
              f"ΔMAFFT: {improvement_from_mafft:>+7.0f} | "
              f"ΔClustal: {improvement_from_clustal:>+7.0f}")
    
    # 8. آنالیز بهبود
    print("\n" + "-" * 40)
    print("IMPROVEMENT ANALYSIS:")
    print("-" * 40)
    
    max_improvement_from_mafft = sp_final - sp_mafft
    max_improvement_from_clustal = sp_final - sp_clustal
    
    if max_improvement_from_mafft > 0:
        print(f"✓ Our method beats MAFFT by {max_improvement_from_mafft:.0f} points ({max_improvement_from_mafft/sp_mafft*100:.1f}%)")
    else:
        print(f"✗ Our method is worse than MAFFT by {abs(max_improvement_from_mafft):.0f} points")
    
    if max_improvement_from_clustal > 0:
        print(f"✓ Our method beats ClustalW by {max_improvement_from_clustal:.0f} points ({max_improvement_from_clustal/sp_clustal*100:.1f}%)")
    else:
        print(f"✗ Our method is worse than ClustalW by {abs(max_improvement_from_clustal):.0f} points")
    
    # 9. ذخیره نتایج
    exp_dir = f"experiments/improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)
    
    from utils.fasta_io import write_fasta
    
    # ذخیره همترازی نهایی
    write_fasta(
        [f">seq{i}" for i in range(len(aln_final))],
        aln_final,
        os.path.join(exp_dir, "final_alignment.fasta")
    )
    
    # ذخیره مقایسه نتایج
    with open(os.path.join(exp_dir, "comparison.txt"), "w") as f:
        f.write("RNA Sequence Alignment Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of sequences: {len(sequences)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SP Scores:\n")
        f.write("-" * 30 + "\n")
        for method, score in sorted_results:
            f.write(f"{method:<15}: {score:>8.0f}\n")
    
    print(f"\n✓ Results saved to: {exp_dir}")
    print("✓ Files created:")
    print(f"  - {exp_dir}/final_alignment.fasta")
    print(f"  - {exp_dir}/comparison.txt")
    
    return aln_final, results

if __name__ == "__main__":
    # اگر فایل ورودی وجود ندارد، یک فایل نمونه بسازید
    if not os.path.exists("input.fasta"):
        print("Warning: input.fasta not found. Creating sample file...")
        create_sample_fasta()
    
    run_improved_pipeline()

def create_sample_fasta():
    """ایجاد یک فایل فاستای نمونه برای تست"""
    sample_sequences = [
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
        for i, seq in enumerate(sample_sequences):
            f.write(f">sequence_{i+1}\n")
            f.write(seq + "\n")
    
    print("Sample FASTA file created: input.fasta")
