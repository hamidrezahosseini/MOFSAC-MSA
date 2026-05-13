# ensemble_strategy.py
import numpy as np
from copy import deepcopy

class EnsembleAlignment:
    """استراتژی Ensemble برای ترکیب بهترین بخش‌های همترازی‌های مختلف"""
    
    def __init__(self, sequences):
        self.sequences = sequences
        
    def create_ensemble(self, alignments):
        """
        ایجاد همترازی ensemble از چندین همترازی ورودی
        
        alignments: لیستی از همترازی‌ها (هر کدام list of strings)
        """
        if not alignments:
            return None
            
        # انتخاب طول همترازی (میانگین طول‌ها)
        lengths = [len(aln[0]) for aln in alignments]
        target_length = int(np.mean(lengths))
        
        n_seq = len(alignments[0])
        
        # ماتریس رای‌گیری برای هر موقعیت
        votes = []
        
        for col in range(target_length):
            position_votes = []
            
            # جمع‌آوری کاراکترها از همه همترازی‌ها
            for seq_idx in range(n_seq):
                chars = []
                for aln in alignments:
                    # اگر این همترازی طول کافی دارد
                    if col < len(aln[0]):
                        chars.append(aln[seq_idx][col])
                    else:
                        chars.append('-')
                
                # رای‌گیری برای محبوب‌ترین کاراکتر
                unique_chars, counts = np.unique(chars, return_counts=True)
                most_common = unique_chars[np.argmax(counts)]
                position_votes.append(most_common)
            
            votes.append(position_votes)
        
        # ساخت همترازی نهایی
        ensemble_alignment = []
        for seq_idx in range(n_seq):
            seq_chars = [votes[col][seq_idx] for col in range(target_length)]
            ensemble_alignment.append(''.join(seq_chars))
        
        return ensemble_alignment
    
    def column_selection(self, alignments):
        """
        انتخاب ستون‌به‌ستون بهترین ستون از بین همترازی‌ها
        """
        n_seq = len(alignments[0])
        
        # ارزیابی کیفیت هر ستون در هر همترازی
        all_columns = []
        column_scores = []
        
        for aln in alignments:
            L = len(aln[0])
            for col in range(L):
                column = [aln[i][col] for i in range(n_seq)]
                all_columns.append(column)
                
                # امتیازدهی به ستون
                score = self._score_column(column)
                column_scores.append((score, col, aln))
        
        # مرتب‌سازی ستون‌ها بر اساس امتیاز
        column_scores.sort(reverse=True, key=lambda x: x[0])
        
        # انتخاب بهترین ستون‌ها (حذف ستون‌های تکراری)
        selected_columns = []
        selected_col_indices = set()
        
        for score, col_idx, aln in column_scores:
            if col_idx not in selected_col_indices:
                column = [aln[i][col_idx] for i in range(n_seq)]
                selected_columns.append(column)
                selected_col_indices.add(col_idx)
        
        # ساخت همترازی نهایی
        final_alignment = []
        for seq_idx in range(n_seq):
            seq_chars = [col[seq_idx] for col in selected_columns]
            final_alignment.append(''.join(seq_chars))
        
        return final_alignment
    
    def _score_column(self, column):
        """امتیازدهی به یک ستون"""
        # پاداش برای ستون‌های بدون شکاف
        if '-' not in column:
            return 10
        
        # پاداش برای ستون‌های با شکاف کم
        gap_ratio = column.count('-') / len(column)
        if gap_ratio < 0.2:
            return 8 - gap_ratio * 10
        
        # جریمه برای ستون‌های با شکاف زیاد
        return 2 - gap_ratio * 10
    
    def hybrid_ensemble(self, mafft_aln, clustal_aln, eo_aln, hybrid_aln):
        """
        ترکیب هوشمندانه چندین همترازی
        """
        # ارزیابی کیفیت هر همترازی
        from tools.msa_scoring import sp_score
        scores = {
            'mafft': sp_score(mafft_aln),
            'clustal': sp_score(clustal_aln),
            'eo': sp_score(eo_aln),
            'hybrid': sp_score(hybrid_aln)
        }
        
        # انتخاب دو همترازی برتر
        sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_methods = [sorted_methods[0][0], sorted_methods[1][0]]
        
        # نگاشت نام متد به همترازی
        aln_map = {
            'mafft': mafft_aln,
            'clustal': clustal_aln,
            'eo': eo_aln,
            'hybrid': hybrid_aln
        }
        
        # ترکیب دو همترازی برتر
        alignments_to_combine = [aln_map[m] for m in top_methods]
        ensemble_result = self.create_ensemble(alignments_to_combine)
        
        return ensemble_result