# hybrid_strategy.py
import numpy as np
from copy import deepcopy
import sys
import os

# اضافه کردن مسیر پروژه به sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import توابع مورد نیاز
from tools.external_aligners import run_mafft, run_clustalw
from tools.msa_scoring import sp_score
from tools.local_refinement import local_refine_alignment
from representation.hybrid_pro import build_hybrid_spec, decode_hybrid
from eo.equilibrium_optimizer_pro import equilibrium_optimizer_pro

class HybridStrategy:
    def __init__(self, sequences):
        self.sequences = sequences
        
    def align(self):
        """استراتژی ترکیبی برای همترازی"""
        
        # 1. مرحله اول: استفاده از MAFFT برای همترازی اولیه
        print("  Step 1: Running MAFFT...")
        aln_mafft = run_mafft(self.sequences)
        
        # 2. مرحله دوم: بهبود با EO-Pro
        print("  Step 2: Improving with EO-Pro...")
        improved = self.improve_with_eo(aln_mafft)
        
        # 3. مرحله سوم: اصلاح محلی
        print("  Step 3: Applying local refinement...")
        refined = local_refine_alignment(improved, max_iters=100)
        
        # 4. مرحله چهارم: ترکیب با ClustalW اگر بهتر باشد
        print("  Step 4: Comparing with ClustalW...")
        aln_clustal = run_clustalw(self.sequences)
        if sp_score(aln_clustal) > sp_score(refined):
            print("  ClustalW is better, using it as final alignment")
            refined = aln_clustal
            
        return refined
    
    def improve_with_eo(self, seed_alignment):
        """بهبود همترازی با EO-Pro"""
        
        # استفاده از تنظیمات ساده‌تر برای اجرای سریع‌تر
        spec = build_hybrid_spec(
            seed_alignment,
            K_insert=4,  # کاهش تعداد insertions
            K_delete=4,  # کاهش تعداد deletions
            M_swaps=3,   # کاهش تعداد swaps
            S_segments=4 # کاهش تعداد segments
        )
        
        # تابع fitness بهبودیافته
        def enhanced_fitness(alignment):
            sp = sp_score(alignment)
            
            # جریمه برای شکاف‌های زیاد
            total_gaps = sum(s.count('-') for s in alignment)
            gap_penalty = total_gaps * 0.05
            
            # پاداش برای ستون‌های مشابه
            similarity_bonus = self._calculate_similarity_bonus(alignment)
            
            return sp - gap_penalty + similarity_bonus, {"sp": sp}
        
        # اجرای EO با پارامترهای بهینه برای اجرای سریع
        print("    Running EO-Pro optimization...")
        result = equilibrium_optimizer_pro(
            decode_fn=lambda v, s, max_shift=20: decode_hybrid(seed_alignment, v, spec, max_shift),
            fitness_fn=enhanced_fitness,
            spec=spec,
            pop_size=30,     # کاهش برای اجرای سریع‌تر
            max_iter=50,     # کاهش برای اجرای سریع‌تر
            low=-3.0,
            high=3.0,
            eq_pool_size=3,
            p_mut=0.15,
            p_reseed=0.03,
            levy_scale=0.02,
            verbose=False
        )
        
        return result["best_alignment"]
    
    def _calculate_similarity_bonus(self, alignment):
        """محاسبه پاداش شباهت"""
        n = len(alignment)
        if n == 0:
            return 0
            
        L = len(alignment[0])
        bonus = 0
        
        for col in range(L):
            chars = [alignment[i][col] for i in range(n)]
            # تعداد کاراکترهای غیر از شکاف
            non_gap_chars = [c for c in chars if c != '-']
            
            if not non_gap_chars:
                continue
                
            # محاسبه شباهت
            most_common = max(set(non_gap_chars), key=non_gap_chars.count)
            similarity_ratio = non_gap_chars.count(most_common) / len(non_gap_chars)
            
            if similarity_ratio == 1.0:  # همه یکسان
                bonus += 5
            elif similarity_ratio >= 0.8:  # حداقل 80% مشابه
                bonus += 2
                
        return bonus