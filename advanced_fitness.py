# advanced_fitness.py
import numpy as np
from scoring.spscore import sp_score


class AdvancedFitness:
    """
    تابع fitness پیشرفته با معیارهای چندگانه
    """

    def __init__(self, match=2, mismatch=-1, gap=-2):
        self.match = match
        self.mismatch = mismatch
        self.gap = gap

    def calculate(self, alignment):
        """
        محاسبه fitness با ترکیب چندین معیار:
        1. SP Score (60%)
        2. Column Conservation (20%)
        3. Gap Penalty (10%)
        4. Alignment Length Penalty (10%)
        """
        n = len(alignment)
        L = len(alignment[0])

        # 1. SP Score
        sp = sp_score(alignment, self.match, self.mismatch, self.gap)

        # 2. Column Conservation Score
        conservation_score = 0
        for col in range(L):
            col_chars = [alignment[i][col] for i in range(n)]
            non_gap_chars = [c for c in col_chars if c != '-']

            if len(non_gap_chars) > 1:
                unique_chars = set(non_gap_chars)
                if len(unique_chars) == 1:
                    conservation_score += 5
                elif len(unique_chars) <= max(2, len(non_gap_chars) // 2):
                    conservation_score += 2

        # 3. Gap Penalty
        total_gaps = sum(seq.count('-') for seq in alignment)
        gap_penalty = total_gaps * 0.1

        # 4. Alignment Length Penalty
        length_penalty = L * 0.01

        # ترکیب وزنی
        total_score = (
            0.6 * sp +
            0.2 * conservation_score -
            0.1 * gap_penalty -
            0.1 * length_penalty
        )

        return total_score, {
            "sp": sp,
            "conservation": conservation_score,
            "total_gaps": total_gaps,
            "length": L
        }


class SPOnlyFitness:
    """
    Fitness ساده فقط بر اساس SP score
    (برای Ablation Study)
    """

    def __init__(self, match=2, mismatch=-1, gap=-2):
        self.match = match
        self.mismatch = mismatch
        self.gap = gap

    def calculate(self, alignment):
        sp = sp_score(alignment, self.match, self.mismatch, self.gap)
        return sp, {
            "sp": sp
        }
