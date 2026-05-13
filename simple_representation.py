# simple_representation.py
import numpy as np

class DisplacementDecoder:
    """
    مسئول تبدیل بردار جابجایی (Continuous Vector) به Alignment نهایی
    """
    def __init__(self, seed_alignment):
        self.seed_alignment = seed_alignment
        self.sequences = [seq.replace('-', '') for seq in seed_alignment]
        self.num_seqs = len(self.sequences)
        
        # تعیین بعد (Dimension): تعداد کل گپ‌های موجود در Seed
        self.total_gaps = sum(seq.count('-') for seq in seed_alignment)
        
        # اگر در Seed گپ نباشد، یک بعد پیش‌فرض در نظر می‌گیریم
        self.dim = max(self.total_gaps, self.num_seqs * 2)

    def decode(self, vector):
        """
        تبدیل بردار عددی به یک Alignment معتبر
        """
        # نرمال‌سازی بردار (اختیاری - بسته به بازه EO)
        # فرض بر این است که بردار شامل مقادیر جابجایی برای گپ‌هاست
        
        alignment = []
        # یک منطق جابجایی ساده (Displacement Logic):
        # برای هر سکانس، مقداری گپ بر اساس مقادیر بردار توزیع می‌شود
        
        chunk_size = len(vector) // self.num_seqs
        
        for i in range(self.num_seqs):
            seq = self.sequences[i]
            seq_vec = vector[i*chunk_size : (i+1)*chunk_size]
            
            # تبدیل مقادیر اعشاری به موقعیت‌های گپ
            # این بخش قلب Displacement Representation است
            new_seq = self._apply_displacement(seq, seq_vec)
            alignment.append(new_seq)
            
        # یکسان‌سازی طول سکانس‌ها با اضافه کردن گپ به انتها (Padding)
        max_len = max(len(s) for s in alignment)
        padded_alignment = [s.ljust(max_len, '-') for s in alignment]
        
        return padded_alignment

    def _apply_displacement(self, seq, vec):
        """
        منطق داخلی برای تزریق گپ به سکانس بر اساس بردار عددی
        """
        chars = list(seq)
        # به ازای هر مقدار مثبت در بردار، در یک موقعیت تصادفی/محاسباتی گپ اضافه می‌کنیم
        # این یک پیاده‌سازی Baseline است که EO آن را بهینه می‌کند
        for val in vec:
            if val > 0.5: # حد آستانه برای ایجاد گپ
                pos = int(abs(val * 100) % (len(chars) + 1))
                chars.insert(pos, '-')
        return "".join(chars)
