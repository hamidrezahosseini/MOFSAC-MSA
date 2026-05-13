"""
محاسبه معیارهای Column Score (CS) و Total Column (TC)
"""

def calculate_column_score(aligned_sequences):
    """
    محاسبه Column Score (CS)
    
    CS: نسبت ستون‌هایی که به طور کامل مطابقت دارند (بدون در نظر گرفتن گپ‌ها)
    """
    if not aligned_sequences or len(aligned_sequences) < 2:
        return 0.0
    
    n_seq = len(aligned_sequences)
    aln_len = len(aligned_sequences[0])
    
    perfect_columns = 0
    
    for col in range(aln_len):
        column_chars = [seq[col] for seq in aligned_sequences]
        # حذف گپ‌ها
        non_gap_chars = [char for char in column_chars if char != '-']
        
        if not non_gap_chars:
            continue  # اگر همه گپ باشند
        
        # بررسی اینکه آیا همه کاراکترهای غیرگپ یکسان هستند
        if all(char == non_gap_chars[0] for char in non_gap_chars):
            perfect_columns += 1
    
    return perfect_columns / aln_len if aln_len > 0 else 0.0


def calculate_total_column_score(aligned_sequences, reference_alignment):
    """
    محاسبه Total Column Score (TC)
    
    TC: نسبت ستون‌هایی که دقیقاً با تراز مرجع مطابقت دارند
    """
    if not aligned_sequences or not reference_alignment:
        return 0.0
    
    if len(aligned_sequences) != len(reference_alignment):
        raise ValueError("تعداد توالی‌ها در تراز و مرجع یکسان نیست")
    
    aln_len = len(aligned_sequences[0])
    if aln_len != len(reference_alignment[0]):
        raise ValueError("طول تراز و مرجع یکسان نیست")
    
    perfect_columns = 0
    
    for col in range(aln_len):
        match = True
        for i in range(len(aligned_sequences)):
            if aligned_sequences[i][col] != reference_alignment[i][col]:
                match = False
                break
        
        if match:
            perfect_columns += 1
    
    return perfect_columns / aln_len if aln_len > 0 else 0.0


def calculate_accuracy(aligned_sequences, reference_alignment=None):
    """
    محاسبه دقت کلی همترازی
    """
    if not aligned_sequences:
        return 0.0
    
    n_seq = len(aligned_sequences)
    aln_len = len(aligned_sequences[0])
    
    if reference_alignment:
        # محاسبه بر اساس مرجع
        return calculate_total_column_score(aligned_sequences, reference_alignment) * 100
    
    # محاسبه تقریبی بر اساس conservation
    conserved_positions = 0
    total_positions = n_seq * aln_len
    
    for col in range(aln_len):
        column_chars = [seq[col] for seq in aligned_sequences]
        # حذف گپ‌ها
        non_gap_chars = [char for char in column_chars if char != '-']
        
        if not non_gap_chars:
            continue
        
        # محاسبه conservation
        most_common = max(set(non_gap_chars), key=non_gap_chars.count)
        conserved_count = non_gap_chars.count(most_common)
        conserved_positions += conserved_count
    
    return (conserved_positions / total_positions * 100) if total_positions > 0 else 0.0


def calculate_all_metrics(aligned_sequences, reference_alignment=None):
    """
    محاسبه تمام معیارهای ارزیابی
    """
    from scoring.spscore import sp_score
    
    metrics = {}
    
    # SP Score
    metrics['SP'] = sp_score(aligned_sequences)
    
    # CS Score
    metrics['CS'] = calculate_column_score(aligned_sequences)
    
    # TC Score (اگر مرجع وجود داشته باشد)
    if reference_alignment:
        metrics['TC'] = calculate_total_column_score(aligned_sequences, reference_alignment)
    
    # Accuracy
    metrics['accuracy'] = calculate_accuracy(aligned_sequences, reference_alignment)
    
    # Gap statistics
    metrics['gap_count'] = sum(seq.count('-') for seq in aligned_sequences)
    metrics['gap_percentage'] = (metrics['gap_count'] / (len(aligned_sequences) * len(aligned_sequences[0])) * 100) if aligned_sequences else 0
    
    return metrics