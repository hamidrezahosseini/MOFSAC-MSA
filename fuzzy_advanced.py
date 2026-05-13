# fuzzy_advanced.py
import numpy as np
from scoring.spscore import sp_score

class AdvancedFuzzyFitness:
    """تابع fitness پیشرفته بر اساس منطق فازی چندهدفه مقاله"""
    
    def __init__(self, match=2, mismatch=-1, gap=-2):
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        
    def calculate(self, alignment):
        """
        محاسبه fitness بر اساس ۴ هدف:
        1. کمینه کردن mismatches
        2. کمینه کردن gap penalties
        3. بیشینه کردن alignment score
        4. کمینه کردن computational cost (تعداد شکاف)
        """
        n = len(alignment)
        L = len(alignment[0])
        
        # 1. محاسبه mismatches
        mismatch_count = 0
        for col in range(L):
            for i in range(n):
                for j in range(i+1, n):
                    if (alignment[i][col] != '-' and alignment[j][col] != '-' and 
                        alignment[i][col] != alignment[j][col]):
                        mismatch_count += 1
        
        # 2. محاسبه gap penalties
        gap_penalty = 0
        for col in range(L):
            for i in range(n):
                for j in range(i+1, n):
                    if alignment[i][col] == '-' or alignment[j][col] == '-':
                        gap_penalty += 1
        
        # 3. محاسبه alignment score
        alignment_score = sp_score(alignment, self.match, self.mismatch, self.gap)
        
        # 4. محاسبه computational cost (تعداد کل شکاف‌ها)
        total_gaps = sum(seq.count('-') for seq in alignment)
        
        # توابع عضویت فازی
        mu_mismatch = self._decreasing_membership(mismatch_count, 0, n*(n-1)//2 * L * 0.3)
        mu_gap = self._decreasing_membership(gap_penalty, 0, n*(n-1)//2 * L * 0.4)
        mu_score = self._increasing_membership(alignment_score, 
                                              n*(n-1)//2 * L * self.mismatch,
                                              n*(n-1)//2 * L * self.match)
        mu_gap_count = self._decreasing_membership(total_gaps, 0, n*L*0.5)
        
        # وزن‌ها
        weights = [0.25, 0.25, 0.35, 0.15]
        fuzzy_score = (weights[0] * mu_mismatch +
                      weights[1] * mu_gap +
                      weights[2] * mu_score +
                      weights[3] * mu_gap_count)
        
        # ترکیب با SP score
        combined_score = fuzzy_score * 100000 + alignment_score
        
        return combined_score, {
            "sp": alignment_score,
            "fuzzy": fuzzy_score,
            "mismatches": mismatch_count,
            "gap_penalty": gap_penalty,
            "total_gaps": total_gaps
        }
    
    def _decreasing_membership(self, x, a, b):
        if x <= a:
            return 1.0
        elif x >= b:
            return 0.0
        else:
            return 1.0 - (x - a) / (b - a)
    
    def _increasing_membership(self, x, a, b):
        if x <= a:
            return 0.0
        elif x >= b:
            return 1.0
        else:
            return (x - a) / (b - a)