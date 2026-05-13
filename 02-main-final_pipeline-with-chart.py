"""
پایپ‌لاین نهایی بهبودیافته برای همترازی توالی‌های RNA با ارزیابی جامع
"""
import os
import time
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Import توابع پایه
from utils.fasta_io import read_fasta, write_fasta
from tools.external_aligners import run_mafft, run_clustalw
from tools.msa_scoring import sp_score
from tools.local_refinement import local_refine_alignment

# Import ماژول‌های جدید
from representation.hybrid_pro import build_hybrid_spec
from eo.equilibrium_optimizer_pro import equilibrium_optimizer_pro
from scoring.advanced_metrics import compute_all_metrics
from visualization import AlignmentVisualizer


def run_final_pipeline(fasta_path="input.fasta", use_rl_params=False, verbose=True):
    """پایپ‌لاین نهایی بهبودیافته"""
    
    if verbose:
        print("="*70)
        print("FINAL IMPROVED PIPELINE FOR RNA SEQUENCE ALIGNMENT")
        print("="*70)
    
    # 1. بارگذاری داده‌ها
    if verbose:
        print("\n📂 Loading sequences...")
    names, sequences = read_fasta(fasta_path)
    if verbose:
        print(f"   ✓ Loaded {len(sequences)} sequences")
        print(f"   Sequence length range: {min(len(s) for s in sequences)}-{max(len(s) for s in sequences)}")
    
    # 2. روش‌های پایه
    if verbose:
        print("\n" + "-"*70)
        print("STEP 1: BASELINE METHODS")
        print("-"*70)
    
    # MAFFT
    if verbose:
        print("\n🔧 Running MAFFT...")
    start = time.time()
    try:
        aln_mafft = run_mafft(sequences)
        mafft_time = time.time() - start
        sp_mafft = sp_score(aln_mafft)
        if verbose:
            print(f"   ✓ MAFFT: SP={sp_mafft:.0f}, Time={mafft_time:.1f}s")
    except Exception as e:
        if verbose:
            print(f"   ✗ MAFFT failed: {e}")
        aln_mafft = sequences
        mafft_time = 0
        sp_mafft = sp_score(aln_mafft)
    
    # ClustalW
    if verbose:
        print("\n🔧 Running ClustalW...")
    start = time.time()
    try:
        aln_clustal = run_clustalw(sequences)
        clustal_time = time.time() - start
        sp_clustal = sp_score(aln_clustal)
        if verbose:
            print(f"   ✓ ClustalW: SP={sp_clustal:.0f}, Time={clustal_time:.1f}s")
    except Exception as e:
        if verbose:
            print(f"   ✗ ClustalW failed: {e}")
        aln_clustal = sequences
        clustal_time = 0
        sp_clustal = sp_score(aln_clustal)
    
    # انتخاب seed
    if sp_clustal >= sp_mafft:
        seed_alignment = aln_clustal
        seed_sp = sp_clustal
        seed_name = "ClustalW"
    else:
        seed_alignment = aln_mafft
        seed_sp = sp_mafft
        seed_name = "MAFFT"
    
    if verbose:
        print(f"\n🎯 Selected seed: {seed_name} (SP={seed_sp:.0f})")
    
    # 3. آماده‌سازی برای EO
    if verbose:
        print("\n" + "-"*70)
        print("STEP 2: PREPARING FOR OPTIMIZATION")
        print("-"*70)
    
    if verbose:
        print("\n📊 Creating hybrid specification...")
    try:
        spec = build_hybrid_spec(
            seed_alignment,
            K_insert=6,
            K_delete=6,
            M_swaps=4,
            S_segments=6
        )
        if verbose:
            print(f"   ✓ Specification created with dimension: {spec['dim']}")
    except Exception as e:
        if verbose:
            print(f"   ✗ Failed to create spec: {e}")
        return None, None
    
    # 4. اجرای EO بهینه‌سازی شده
    if verbose:
        print("\n" + "-"*70)
        print("STEP 3: OPTIMIZED EO ALGORITHM")
        print("-"*70)
    
    if verbose:
        print("\n🚀 Running EO-Pro optimization...")
    
    # تابع decode wrapper
    def decode_wrapper(v, spec_in, max_shift=20):
        from representation.hybrid_pro import decode_hybrid
        return decode_hybrid(seed_alignment, v, spec_in, max_shift=max_shift)
    
    # تابع fitness
    def fitness_fn(alignment):
        score = sp_score(alignment)
        # جریمه برای شکاف‌های زیاد
        total_gaps = sum(s.count('-') for s in alignment)
        gap_penalty = total_gaps * 0.01
        return score - gap_penalty, {"sp": score, "gaps": total_gaps}
    
    start = time.time()
    
    # پارامترهای EO (در صورت استفاده از RL، می‌توانند متفاوت باشند)
    if use_rl_params:
        # پارامترهای بهینه از RL
        eo_params = {
            'pop_size': 40,
            'max_iter': 60,
            'low': -3.0,
            'high': 3.0,
            'eq_pool_size': 4,
            'p_mut': 0.12,
            'p_reseed': 0.03,
            'levy_scale': 0.04
        }
        method_name = "EO-Pro (RL)"
    else:
        # پارامترهای پیش‌فرض
        eo_params = {
            'pop_size': 35,
            'max_iter': 50,
            'low': -3.0,
            'high': 3.0,
            'eq_pool_size': 4,
            'p_mut': 0.15,
            'p_reseed': 0.02,
            'levy_scale': 0.03
        }
        method_name = "EO-Pro (Default)"
    
    if verbose:
        print(f"   Parameters: {eo_params}")
    
    try:
        eo_result = equilibrium_optimizer_pro(
            decode_fn=decode_wrapper,
            fitness_fn=fitness_fn,
            spec=spec,
            **eo_params,
            verbose=False,
            seed=42
        )
        eo_time = time.time() - start
        
        aln_eo = eo_result["best_alignment"]
        eo_fitness = eo_result["best_fitness"]
        sp_eo = sp_score(aln_eo)
        
        if verbose:
            print(f"\n✅ {method_name} completed:")
            print(f"   - Final fitness: {eo_fitness:.1f}")
            print(f"   - SP score: {sp_eo:.0f}")
            print(f"   - Improvement from seed: {sp_eo - seed_sp:+.0f}")
            print(f"   - Time: {eo_time:.1f}s")
    except Exception as e:
        if verbose:
            print(f"   ✗ EO optimization failed: {e}")
        # در صورت شکست، از seed استفاده می‌کنیم
        aln_eo = seed_alignment
        eo_time = 0
        sp_eo = seed_sp
    
    # 5. اصلاح محلی
    if verbose:
        print("\n" + "-"*70)
        print("STEP 4: LOCAL REFINEMENT")
        print("-"*70)
    
    if verbose:
        print("\n🔧 Applying local refinement...")
    start_refine = time.time()
    try:
        aln_refined = local_refine_alignment(aln_eo, max_iters=50)
        refine_time = time.time() - start_refine
        sp_refined = sp_score(aln_refined)
        
        # اگر اصلاح محلی باعث بهبود نشد، از نسخه EO استفاده می‌کنیم
        if sp_refined < sp_eo:
            aln_refined = aln_eo
            sp_refined = sp_eo
            refine_time = 0
            
        if verbose:
            print(f"   ✓ Refinement completed in {refine_time:.1f}s")
            print(f"   - SP after refinement: {sp_refined:.0f}")
            if sp_refined > sp_eo:
                print(f"   - Improvement from EO: +{sp_refined - sp_eo:.0f}")
    except Exception as e:
        if verbose:
            print(f"   ✗ Local refinement failed: {e}")
        aln_refined = aln_eo
        sp_refined = sp_eo
        refine_time = 0
    
    # 6. محاسبه معیارهای پیشرفته برای همه روش‌ها
    if verbose:
        print("\n" + "-"*70)
        print("STEP 5: ADVANCED METRICS CALCULATION")
        print("-"*70)
    
    if verbose:
        print("\n📊 Calculating advanced metrics for all methods...")
    
    all_metrics = {}
    
    # محاسبه معیارها برای MAFFT
    try:
        mafft_metrics = compute_all_metrics(aln_mafft, "MAFFT", mafft_time)
        all_metrics["MAFFT"] = mafft_metrics
        if verbose:
            print(f"   ✓ MAFFT: SP={mafft_metrics['sp_score']:.0f}, TC={mafft_metrics['tc_score']:.3f}, Time={mafft_time:.1f}s")
    except Exception as e:
        if verbose:
            print(f"   ✗ MAFFT metrics failed: {e}")
    
    # محاسبه معیارها برای ClustalW
    try:
        clustal_metrics = compute_all_metrics(aln_clustal, "ClustalW", clustal_time)
        all_metrics["ClustalW"] = clustal_metrics
        if verbose:
            print(f"   ✓ ClustalW: SP={clustal_metrics['sp_score']:.0f}, TC={clustal_metrics['tc_score']:.3f}, Time={clustal_time:.1f}s")
    except Exception as e:
        if verbose:
            print(f"   ✗ ClustalW metrics failed: {e}")
    
    # محاسبه معیارها برای EO
    try:
        eo_metrics = compute_all_metrics(aln_eo, method_name, eo_time)
        all_metrics[method_name] = eo_metrics
        if verbose:
            print(f"   ✓ {method_name}: SP={eo_metrics['sp_score']:.0f}, TC={eo_metrics['tc_score']:.3f}, Time={eo_time:.1f}s")
    except Exception as e:
        if verbose:
            print(f"   ✗ {method_name} metrics failed: {e}")
    
    # محاسبه معیارها برای Final Refined
    try:
        final_metrics = compute_all_metrics(aln_refined, "Final_Refined", eo_time + refine_time)
        all_metrics["Final_Refined"] = final_metrics
        if verbose:
            print(f"   ✓ Final Refined: SP={final_metrics['sp_score']:.0f}, TC={final_metrics['tc_score']:.3f}, Time={(eo_time + refine_time):.1f}s")
    except Exception as e:
        if verbose:
            print(f"   ✗ Final Refined metrics failed: {e}")
    
    # 7. نمایش جدول مقایسه جامع
    if verbose:
        print("\n" + "-"*70)
        print("COMPREHENSIVE COMPARISON TABLE")
        print("-"*70)
    
    # ایجاد DataFrame برای نمایش
    comparison_data = []
    for method_name_clean, metrics in all_metrics.items():
        row = {
            'Method': method_name_clean,
            'SP Score': f"{metrics['sp_score']:.0f}",
            'TC Score': f"{metrics['tc_score']:.3f}",
            'CS Score': f"{metrics['cs_score']:.3f}",
            'Conservation': f"{metrics['conservation_score']:.3f}",
            'Gap %': f"{metrics['gap_percentage']:.1f}%",
            'Time (s)': f"{metrics['execution_time']:.2f}",
            'Avg Identity': f"{metrics['avg_identity']:.1f}%",
            'Length': metrics['alignment_length'],
            'Sequences': metrics['num_sequences']
        }
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # اضافه کردن ستون Improvement (نسبت به بهترین روش پایه)
    base_methods = ['MAFFT', 'ClustalW']
    base_sp_scores = [all_metrics.get(m, {}).get('sp_score', 0) for m in base_methods if m in all_metrics]
    
    if base_sp_scores:
        best_base_sp = max(base_sp_scores)
        improvements = []
        for method_name_clean in df_comparison['Method']:
            if method_name_clean in all_metrics:
                current_sp = all_metrics[method_name_clean]['sp_score']
                if method_name_clean in base_methods:
                    improvements.append("0.0% (baseline)")
                else:
                    improvement_pct = ((current_sp - best_base_sp) / best_base_sp * 100) if best_base_sp > 0 else 0
                    improvements.append(f"{improvement_pct:+.1f}%")
            else:
                improvements.append("N/A")
        
        df_comparison['Improvement'] = improvements
    
    # مرتب‌سازی بر اساس SP Score
    df_comparison['SP_Value'] = df_comparison['SP Score'].astype(float)
    df_comparison = df_comparison.sort_values('SP_Value', ascending=False).drop('SP_Value', axis=1)
    
    if verbose:
        print("\n" + df_comparison.to_string(index=False))
    
    # 8. ذخیره نتایج
    if verbose:
        print("\n" + "-"*70)
        print("STEP 6: SAVING RESULTS")
        print("-"*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"results/final_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    if verbose:
        print(f"\n📁 Creating experiment directory: {exp_dir}")
    
    # ذخیره همترازی‌ها
    alignments = {
        "mafft": aln_mafft,
        "clustal": aln_clustal,
        "eo": aln_eo,
        "final": aln_refined
    }
    
    for name, aln in alignments.items():
        path = os.path.join(exp_dir, f"{name}_alignment.fasta")
        try:
            write_fasta([f">seq_{i}" for i in range(len(aln))], aln, path)
            if verbose:
                print(f"   ✓ Saved {name} alignment: {path}")
        except Exception as e:
            if verbose:
                print(f"   ✗ Failed to save {name} alignment: {e}")
    
    # 9. ایجاد نمودارهای مقایسه
    if verbose:
        print("\n📈 Generating comparison plots...")
    
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        visualizer = AlignmentVisualizer()
        
        # نمودار جامع مقایسه
        fig1 = visualizer.plot_comprehensive_comparison(
            all_metrics,
            save_path=os.path.join(plots_dir, "comprehensive_comparison.png"),
            show_plot=False
        )
        
        # نمودار رادار
        fig2 = visualizer.plot_radar_chart(
            all_metrics,
            save_path=os.path.join(plots_dir, "radar_comparison.png"),
            show_plot=False
        )
        
        # تحلیل trade-off
        fig3 = visualizer.plot_tradeoff_analysis(
            all_metrics,
            save_path=os.path.join(plots_dir, "tradeoff_analysis.png"),
            show_plot=False
        )
        
        if verbose:
            print(f"   ✓ Generated 3 comparison plots in: {plots_dir}")
    except Exception as e:
        if verbose:
            print(f"   ✗ Failed to generate plots: {e}")
    
    # 10. ایجاد گزارش HTML
    if verbose:
        print("\n📄 Generating HTML report...")
    
    try:
        html_report_path = os.path.join(exp_dir, "detailed_report.html")
        visualizer.generate_html_report(all_metrics, save_path=html_report_path)
        if verbose:
            print(f"   ✓ HTML report saved: {html_report_path}")
    except Exception as e:
        if verbose:
            print(f"   ✗ Failed to generate HTML report: {e}")
    
    # 11. ذخیره معیارها در JSON
    if verbose:
        print("\n💾 Saving all metrics to JSON...")
    
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
        
        metrics_json_path = os.path.join(exp_dir, "all_metrics.json")
        with open(metrics_json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"   ✓ Metrics saved: {metrics_json_path}")
    except Exception as e:
        if verbose:
            print(f"   ✗ Failed to save metrics: {e}")
    
    # 12. خلاصه نتایج
    if verbose:
        print("\n" + "-"*70)
        print("RESULTS SUMMARY")
        print("-"*70)
        
        # پیدا کردن بهترین روش برای هر معیار
        print("\n🏆 BEST METHOD FOR EACH METRIC:")
        print("-" * 40)
        
        metrics_to_check = ['sp_score', 'tc_score', 'cs_score', 'conservation_score', 
                            'avg_identity', 'execution_time', 'gap_percentage']
        
        metric_names = {
            'sp_score': 'SP Score',
            'tc_score': 'TC Score',
            'cs_score': 'CS Score',
            'conservation_score': 'Conservation',
            'avg_identity': 'Avg Identity',
            'execution_time': 'Execution Time',
            'gap_percentage': 'Gap Percentage'
        }
        
        for metric in metrics_to_check:
            if metric in ['execution_time', 'gap_percentage']:
                # مقادیر کمتر بهتر هستند
                best_val = float('inf')
                best_method = None
                for method, metrics in all_metrics.items():
                    if metric in metrics and metrics[metric] < best_val:
                        best_val = metrics[metric]
                        best_method = method
            else:
                # مقادیر بیشتر بهتر هستند
                best_val = float('-inf')
                best_method = None
                for method, metrics in all_metrics.items():
                    if metric in metrics and metrics[metric] > best_val:
                        best_val = metrics[metric]
                        best_method = method
            
            if best_method:
                # فرمت نمایش
                if metric == 'execution_time':
                    display_val = f"{best_val:.2f}s"
                elif metric == 'gap_percentage':
                    display_val = f"{best_val:.1f}%"
                elif metric == 'avg_identity':
                    display_val = f"{best_val:.1f}%"
                else:
                    display_val = f"{best_val:.3f}"
                
                print(f"  {metric_names[metric]:<20}: {best_method:<20} ({display_val})")
        
        # محاسبه بهبود کلی
        if 'MAFFT' in all_metrics and 'Final_Refined' in all_metrics:
            mafft_sp = all_metrics['MAFFT']['sp_score']
            final_sp = all_metrics['Final_Refined']['sp_score']
            
            if mafft_sp > 0:
                improvement_pct = ((final_sp - mafft_sp) / mafft_sp) * 100
                
                print("\n📈 OVERALL IMPROVEMENT:")
                print("-" * 40)
                print(f"  MAFFT SP Score:      {mafft_sp:.0f}")
                print(f"  Final Refined SP:    {final_sp:.0f}")
                print(f"  Absolute Improvement: {final_sp - mafft_sp:+.0f}")
                print(f"  Percentage Improvement: {improvement_pct:+.1f}%")
                
                if improvement_pct > 0:
                    if improvement_pct > 10:
                        print("\n🎉 EXCELLENT! Significant improvement achieved!")
                    elif improvement_pct > 5:
                        print("\n👏 VERY GOOD! Noticeable improvement achieved!")
                    elif improvement_pct > 0:
                        print("\n👍 GOOD! Small but positive improvement.")
                else:
                    print("\n⚠️  WARNING: No improvement over MAFFT achieved.")
        
        print(f"\n📁 All results saved in: {exp_dir}")
        print(f"📊 HTML Report: {os.path.join(exp_dir, 'detailed_report.html')}")
        print(f"📈 Plots: {plots_dir}/")
        print(f"📋 Metrics: {os.path.join(exp_dir, 'all_metrics.json')}")
    
    return aln_refined, all_metrics, exp_dir


# تابع اصلی اجرا
if __name__ == "__main__":
    print("\n" + "="*70)
    print("RNA SEQUENCE ALIGNMENT PIPELINE")
    print("="*70)
    
    # بررسی وجود فایل ورودی
    if not os.path.exists("input.fasta"):
        print("\n⚠️  input.fasta not found. Creating sample file...")
        
        # توالی‌های RNA نمونه
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
        
        try:
            with open("input.fasta", "w") as f:
                for i, seq in enumerate(sample_seqs):
                    f.write(f">seq_{i+1}\n{seq}\n")
            print("✅ Sample file created: input.fasta")
            print(f"   Contains {len(sample_seqs)} RNA sequences")
        except Exception as e:
            print(f"❌ Failed to create sample file: {e}")
            exit(1)
    
    # اجرای پایپ‌لاین
    try:
        print("\n🚀 Starting pipeline execution...")
        final_alignment, all_metrics, exp_dir = run_final_pipeline(
            fasta_path="input.fasta",
            use_rl_params=False,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # نمایش بهترین روش
        if all_metrics:
            best_method = max(all_metrics.items(), key=lambda x: x[1].get('sp_score', 0))[0]
            best_sp = all_metrics[best_method]['sp_score']
            
            print(f"\n🏆 BEST OVERALL METHOD: {best_method} (SP={best_sp:.0f})")
            print(f"📁 All results saved in: {exp_dir}")
            
            # پیشنهادات
            print("\n💡 RECOMMENDATIONS:")
            if 'Final_Refined' in all_metrics and all_metrics['Final_Refined']['sp_score'] == best_sp:
                print("  • Use the 'Final_Refined' alignment for best accuracy")
            elif 'EO-Pro' in best_method:
                print(f"  • Use '{best_method}' for optimal performance")
            else:
                print(f"  • Use '{best_method}' as it provides the best balance")
            
            print("\n📊 To view detailed results:")
            print(f"  1. Open {os.path.join(exp_dir, 'detailed_report.html')} in your browser")
            print(f"  2. Check the plots in {os.path.join(exp_dir, 'plots')}")
            print(f"  3. Review the metrics in {os.path.join(exp_dir, 'all_metrics.json')}")
        
    except Exception as e:
        print(f"\n❌ ERROR during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 TROUBLESHOOTING:")
        print("  1. Check if MAFFT and ClustalW are installed and paths are correct")
        print("  2. Ensure all required Python packages are installed")
        print("  3. Verify the input.fasta file format is correct")
