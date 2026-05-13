"""
معیارهای پیشرفته برای ارزیابی همترازی چندگانه
شامل SP, TC, CS, Gap Percentage, Conservation Score, و دیگر معیارها
"""
import numpy as np
from scoring.spscore import sp_score

def compute_all_metrics(alignment, method_name="", exec_time=0):
    """
    محاسبه تمام معیارهای ارزیابی برای یک همترازی
    """
    if not alignment or len(alignment) == 0:
        return {}
    
    n_seq = len(alignment)
    aln_len = len(alignment[0])
    
    # بررسی یکسان بودن طول‌ها
    for seq in alignment:
        if len(seq) != aln_len:
            raise ValueError("All sequences must have the same length")
    
    # 1. SP Score (Sum-of-Pairs)
    sp = sp_score(alignment)
    
    # 2. TC Score (Total Column Score)
    tc, perfect_columns = compute_tc_score(alignment)
    
    # 3. CS Score (Column Score) - درصد ستون‌های با conservation بالا
    cs = compute_cs_score(alignment)
    
    # 4. Gap Percentage
    gap_pct = compute_gap_percentage(alignment)
    
    # 5. Conservation Score
    conservation = compute_conservation_score(alignment)
    
    # 6. Shannon Entropy (میانگین آنتروپی ستون‌ها)
    entropy = compute_shannon_entropy(alignment)
    
    # 7. Alignment Length
    aln_length = aln_len
    
    # 8. Average Identity
    avg_identity = compute_average_identity(alignment)
    
    # 9. Compression Ratio (نسبت طول همترازی به میانگین طول توالی‌ها)
    orig_lengths = [len(seq.replace('-', '')) for seq in alignment]
    avg_orig_len = np.mean(orig_lengths) if orig_lengths else aln_len
    compression_ratio = aln_len / avg_orig_len if avg_orig_len > 0 else 1.0
    
    return {
        'method': method_name,
        'sp_score': sp,
        'tc_score': tc,
        'cs_score': cs,
        'gap_percentage': gap_pct,
        'conservation_score': conservation,
        'shannon_entropy': entropy,
        'alignment_length': aln_length,
        'avg_identity': avg_identity,
        'compression_ratio': compression_ratio,
        'execution_time': exec_time,
        'perfect_columns': perfect_columns,
        'num_sequences': n_seq
    }

def compute_tc_score(alignment):
    """
    محاسبه TC Score: درصد ستون‌هایی که کاملاً حفاظت شده‌اند
    (همه کاراکترهای غیر شکاف یکسان هستند)
    """
    n_seq = len(alignment)
    aln_len = len(alignment[0])
    
    perfect_columns = 0
    for col in range(aln_len):
        col_chars = [alignment[i][col] for i in range(n_seq)]
        non_gap_chars = [c for c in col_chars if c != '-']
        
        if len(non_gap_chars) > 0:
            # بررسی آیا همه کاراکترهای غیر شکاف یکسان هستند
            unique_chars = set(non_gap_chars)
            if len(unique_chars) == 1:
                perfect_columns += 1
    
    tc_score = perfect_columns / aln_len if aln_len > 0 else 0
    return tc_score, perfect_columns

def compute_cs_score(alignment, similarity_threshold=0.8):
    """
    محاسبه CS Score: درصد ستون‌هایی با conservation بالا
    (حداقل similarity_threshold از کاراکترها مشابه هستند)
    """
    n_seq = len(alignment)
    aln_len = len(alignment[0])
    
    good_columns = 0
    for col in range(aln_len):
        col_chars = [alignment[i][col] for i in range(n_seq)]
        non_gap_chars = [c for c in col_chars if c != '-']
        
        if len(non_gap_chars) > 0:
            # محاسبه conservation
            from collections import Counter
            counts = Counter(non_gap_chars)
            most_common_count = counts.most_common(1)[0][1]
            similarity = most_common_count / len(non_gap_chars)
            
            if similarity >= similarity_threshold:
                good_columns += 1
    
    cs_score = good_columns / aln_len if aln_len > 0 else 0
    return cs_score

def compute_gap_percentage(alignment):
    """محاسبه درصد شکاف‌ها در کل همترازی"""
    n_seq = len(alignment)
    aln_len = len(alignment[0])
    
    total_chars = n_seq * aln_len
    total_gaps = sum(seq.count('-') for seq in alignment)
    
    gap_pct = (total_gaps / total_chars) * 100 if total_chars > 0 else 0
    return gap_pct

def compute_conservation_score(alignment):
    """
    محاسبه Conservation Score: میانگین شباهت در ستون‌ها
    بازده: 0 (هیچ شباهتی) تا 1 (کاملاً مشابه)
    """
    n_seq = len(alignment)
    aln_len = len(alignment[0])
    
    if n_seq < 2:
        return 0
    
    column_similarities = []
    for col in range(aln_len):
        col_chars = [alignment[i][col] for i in range(n_seq)]
        non_gap_chars = [c for c in col_chars if c != '-']
        
        if len(non_gap_chars) > 1:
            # محاسبه pairwise similarity
            similarity_sum = 0
            pairs = 0
            
            for i in range(len(non_gap_chars)):
                for j in range(i+1, len(non_gap_chars)):
                    if non_gap_chars[i] == non_gap_chars[j]:
                        similarity_sum += 1
                    pairs += 1
            
            column_similarity = similarity_sum / pairs if pairs > 0 else 0
            column_similarities.append(column_similarity)
    
    if not column_similarities:
        return 0
    
    return np.mean(column_similarities)

def compute_shannon_entropy(alignment):
    """محاسبه میانگین آنتروپی شانون برای ستون‌ها"""
    n_seq = len(alignment)
    aln_len = len(alignment[0])
    
    column_entropies = []
    for col in range(aln_len):
        col_chars = [alignment[i][col] for i in range(n_seq)]
        non_gap_chars = [c for c in col_chars if c != '-']
        
        if len(non_gap_chars) > 0:
            # محاسبه فراوانی کاراکترها
            from collections import Counter
            counts = Counter(non_gap_chars)
            total = len(non_gap_chars)
            
            # محاسبه آنتروپی
            entropy = 0
            for count in counts.values():
                p = count / total
                entropy -= p * np.log2(p)
            
            column_entropies.append(entropy)
    
    if not column_entropies:
        return 0
    
    return np.mean(column_entropies)

def compute_average_identity(alignment):
    """محاسبه میانگین identity بین توالی‌ها"""
    n_seq = len(alignment)
    aln_len = len(alignment[0])
    
    if n_seq < 2:
        return 0
    
    identities = []
    for i in range(n_seq):
        for j in range(i+1, n_seq):
            matches = sum(1 for k in range(aln_len) 
                         if alignment[i][k] != '-' and alignment[j][k] != '-' 
                         and alignment[i][k] == alignment[j][k])
            total = sum(1 for k in range(aln_len) 
                       if alignment[i][k] != '-' and alignment[j][k] != '-')
            
            if total > 0:
                identity = matches / total
                identities.append(identity)
    
    if not identities:
        return 0
    
    return np.mean(identities) * 100  # به درصد

# تست سریع
if __name__ == "__main__":
    test_alignment = [
        "AUCG-AUCG",
        "AUCGUAUCG",
        "AUCG-AUCG"
    ]
    metrics = compute_all_metrics(test_alignment, "Test", 1.5)
    for key, value in metrics.items():
        print(f"{key}: {value}")