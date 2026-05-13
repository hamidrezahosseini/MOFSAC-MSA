#gap_aware_decoder.py

import random
import copy

class GapAwareDecoder:
    """
    Decoder که اجازه‌ی درج gap داخلی می‌دهد
    """

    def __init__(self, seed_alignment, max_gaps=3):
        self.seed = seed_alignment
        self.n_seq = len(seed_alignment)
        self.max_gaps = max_gaps

        # هر توالی: موقعیت‌های درج gap
        self.dim = self.n_seq * self.max_gaps

    def decode(self, vector):
        alignment = copy.deepcopy(self.seed)

        idx = 0
        for i in range(self.n_seq):
            for _ in range(self.max_gaps):
                pos = int(abs(vector[idx]) * (len(alignment[i]) + 1)) % (len(alignment[i]) + 1)
                alignment[i] = alignment[i][:pos] + "-" + alignment[i][pos:]
                idx += 1

        # هم‌طول‌سازی
        max_len = max(len(s) for s in alignment)
        alignment = [s.ljust(max_len, "-") for s in alignment]

        return alignment
